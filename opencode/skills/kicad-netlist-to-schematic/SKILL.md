---
name: kicad-netlist-to-schematic
description: Recreate a hierarchical KiCad project (.kicad_pro + .kicad_sch files + SKiDL Python) from a KiCad-eeschema netlist export (.net). Use when the user has a .net file (eeschema s-expression export) and wants to regenerate the schematic / SKiDL source, or asks to "recreate the kicad schematic", "rebuild from netlist", "convert .net to .kicad_sch", "make a SKiDL version of this board", or similar. Also use to debug pin-tip math, label-merging, page sizing, hierarchy/sheet-pin issues in generated KiCad 10 schematics.
---

# KiCad netlist → hierarchical schematic + SKiDL

End-to-end workflow for turning a KiCad **eeschema-exported netlist**
(`.net`, s-expression form) into a re-loadable KiCad project consisting of:

1. **SKiDL Python source** — one module per hierarchical sheet, deterministic, regeneratable.
2. **A KiCad 10 `.kicad_sch` set** — root + one child per hierarchical block, auto-sized pages, sheet pins for cross-sheet nets.
3. **A SKiDL-generated KiCad netlist** (`.skidl.net`) and **a netlist re-exported from the new schematic** — both verified bit-exact against the original.

Verification target throughout: every component preserved, every wire preserved, zero value/footprint drift.

---

## Required inputs

The user only needs to give you:

- `<name>.net` — KiCad eeschema export, s-expression form. Contains `(design ...)`, `(components ...)`, `(libparts ...)`, `(nets ...)` sections.
- (optional) `<name>_BOM.csv` — useful for human cross-check.

The `.net` is the single source of truth. It contains:

- Sheet hierarchy (every `(sheet (name "/Foo/") ...)` under `(design ...)`).
- Every component instance with its `(sheetpath (names "/Foo/"))`.
- Every net with sheet-prefixed name (`/Foo/SCL0`) for hierarchical nets.
- Pin numbers, names, and types for every library part (`(libparts ...)`).

---

## High-level pipeline

```
.net  ──► parse_net.py    ──► netlist.json   (structured dump)
                                    │
                                    ▼
                             build_plan.py   ──► plan.json
                                    │             (per-sheet plan:
                                    │              local nets / hier nets /
                                    │              power rails / wiring)
                                    │
                ┌───────────────────┴───────────────────┐
                ▼                                       ▼
        gen_skidl.py                          gen_kicad_sch.py
                │                                       │
                ▼                                       ▼
     cm4iov5_skidl/  (Python)             *.kicad_sch + .kicad_pro
                │                                       │
                ▼                                       ▼
     CM4IOv5.skidl.net                    CM4IOv5.from_sch.net
                                            (re-exported by kicad-cli)
                │                                       │
                └────────────► diff against original ◄──┘
                              (target: bit-exact)
```

Five Python files, in this order: `parse_net.py`, `build_plan.py`, `gen_skidl.py`, `gen_kicad_sch.py`, `diff_nets.py`.

Reference implementations from a successful run are in
`<project>/cm4iov5_skidl/` after the skill has been applied once.

---

## Step 1 — Parse the netlist

The `.net` is an s-expression file. Write a small tokenizer + recursive parser
(do not pull in `sexpdata` — it mangles quoted strings).

Token rules:
- `(` and `)` are single-char tokens.
- A double-quoted string with backslash escapes is one token (the quotes are NOT part of the value).
- Everything else up to whitespace or paren is a bare symbol.

Emit `netlist.json` containing:

```json
{
  "sheets":     [{"number","name","tstamps","title","source"}, ...],
  "components": [{"ref","value","footprint","lib","part",
                  "sheet_name","sheet_tstamps","properties","fields"}, ...],
  "nets":       [{"code","name","nodes":[{"ref","pin","pintype","pinfunction"}, ...]}, ...],
  "libparts":   [{"lib","part","pins":[{"num","name","type"}, ...]}, ...]
}
```

A typical mid-sized board has 5–10 sheets, 100–250 components, 200–400 nets, 30–50 libparts.

---

## Step 2 — Classify nets and build the plan

Three categories that drive every downstream decision:

| Class       | Definition                                                  | KiCad representation                  |
|-------------|-------------------------------------------------------------|---------------------------------------|
| **power**   | Name is `GND` or `^/?\+?[\d.]+v$` (`+5v`, `+3.3v`, etc.)     | `global_label` + power port symbol    |
| **hier**    | Touches >= 2 sheets and isn't a power rail                  | `global_label` AND `hierarchical_label` on each child; `(pin ...)` on root |
| **local**   | Touches exactly one sheet (named or unnamed `Net-(...)`)    | `label` (local)                       |

Compute `ref_sheet[ref] = sheet_name`, then for each net set
`sheets_touched = {ref_sheet[n.ref] for n in net.nodes}`. Pick class from the
two rules above.

Sanitise net names for Python identifiers (`/CM4_GPIO/SCL0` → `SCL0`).
For display labels on the schematic, **strip the sheet prefix and keep only
the leaf component** so all sheets that share a net use the same label text.
That is what makes global labels merge.

Per-sheet plan entries:

- `refs` — components on the sheet.
- `hier_params` — power + hier nets used by this sheet (passed as subcircuit args in SKiDL, and become sheet pins on the root).
- `local_nets` — local nets created inside the sheet.
- `wiring` — every (net, [(ref, pin), …]) tuple restricted to this sheet's components.

Cross-sheet net naming gotcha: KiCad prefixes hierarchical net names with the
**path of the sheet that first declared them** (e.g. `/CM4_HighSpeed/SCL0`),
so two sheets that share a net see different `name` strings in the netlist —
do not key off the full name; key off the **set of pins** or the leaf name.

---

## Step 3 — Generate SKiDL Python

### Package layout

```
<project>_skidl/
├── __init__.py
├── cm4iov5.py            # top-level
├── parts_lib.py          # one TEMPLATE Part per unique (lib, part)
├── sheets/
│   ├── __init__.py
│   ├── usb2_hub.py
│   └── ...
```

**Critical: do NOT name the package `skidl`** — it shadows the library and
`from skidl import Net` fails. Use `<project>_skidl` or similar.

### `parts_lib.py` — TEMPLATE parts from libparts

Most boards use custom symbol libraries (`CM4IO:ComputeModule4-CM4`, etc.)
that aren't in stock KiCad. Build a `Part(dest=TEMPLATE, ...)` for every
unique `(lib, part)` straight from the netlist's `libparts` section so pin
numbers/names/types match 1:1.

```python
from skidl import Part, Pin, TEMPLATE

_PT = {
    "input": Pin.types.INPUT, "output": Pin.types.OUTPUT,
    "bidirectional": Pin.types.BIDIR, "tri_state": Pin.types.TRISTATE,
    "passive": Pin.types.PASSIVE, "free": Pin.types.FREE,
    "unspecified": Pin.types.UNSPEC, "power_in": Pin.types.PWRIN,
    "power_out": Pin.types.PWROUT, "open_collector": Pin.types.OPENCOLL,
    "open_emitter": Pin.types.OPENEMIT, "no_connect": Pin.types.NOCONNECT,
}

CM4IO__ComputeModule4_CM4 = Part(
    name="ComputeModule4-CM4", dest=TEMPLATE, tool="skidl",
    ref_prefix="Module", footprint="CM4IO:Raspberry-Pi-4-Compute-Module",
    pins=[],
)
CM4IO__ComputeModule4_CM4.pins = [
    Pin(num="1", name="GND", func=_PT["power_in"]),
    # ... one per libpart pin, sorted by integer pin number
]
for _p in CM4IO__ComputeModule4_CM4.pins:
    _p.part = CM4IO__ComputeModule4_CM4    # don't forget this back-link
```

Expose:

```python
TEMPLATES = {("CM4IO", "ComputeModule4-CM4"): CM4IO__ComputeModule4_CM4, ...}

def make(lib, part, ref=None, **kwargs):
    if ref is not None and "tag" not in kwargs:
        kwargs["tag"] = f"<PROJECT>-{ref}"   # silences "Missing tag" warnings
    if ref is not None:
        kwargs["ref"] = ref
    return TEMPLATES[(lib, part)](**kwargs)
```

### Per-sheet subcircuit

```python
from skidl import Net, subcircuit
from ..parts_lib import make

@subcircuit
def rtc_wakeup_fan(GND, P12V, P3V3, P5V, GLOBAL_EN, SCL0, SDA0):
    """Hierarchical nets received from the top level are subcircuit args."""

    # Local nets first
    nRTC_INT = Net("/RTC , Wakeup, FAN/nRTC_INT")    # preserve original name

    # Components grouped by ref-prefix (C, R, L, D, U, J, Module...)
    u8 = make("Timer_RTC", "PCF8563T", ref="U8", value="PCF85063AT/AAZ",
              footprint="Package_SO:SOIC-8_3.9x4.9mm_P1.27mm")
    # ...

    # Wire each pin: `comp[pin] += net`
    u8[4] += GND
    u8[5] += SDA0
    u8[6] += SCL0
```

### Top-level

```python
from skidl import Net, generate_netlist, set_default_tool, KICAD
set_default_tool(KICAD)

def build():
    GND  = Net("GND");  GND.drive = 7    # mark as POWER drive
    P5V  = Net("/+5v")
    P3V3 = Net("/+3.3v")
    P12V = Net("/+12v")

    # Cross-sheet hier nets, declared once at top level
    SCL0 = Net("/CM4_HighSpeed/SCL0")
    # ...

    rtc_wakeup_fan(tag="rtc_wakeup_fan",
                   GND=GND, P12V=P12V, P3V3=P3V3, P5V=P5V,
                   GLOBAL_EN=GLOBAL_EN, SCL0=SCL0, SDA0=SDA0)
    # ... other subcircuits

def main(run_erc=False, netlist_file="<PROJECT>.skidl.net"):
    build()
    if run_erc: ERC()
    generate_netlist(file_=netlist_file)

if __name__ == "__main__":
    import sys
    main(run_erc="--erc" in sys.argv)
```

### Common SKiDL pitfalls

- **Double `if __name__ == "__main__":`** at end of generated file → `main()` runs twice → every part gets a `_1` doppelgänger in the output netlist. Always grep your generator for duplicate `__main__` blocks if you see duplicated refs in the output.
- **Single `import` of the package module re-executes it on each run**. SKiDL is module-level stateful; don't `import` and then call `main()` twice from another script — produces the same doubling.
- **ERC defaults to on**. Many real designs (e.g. the CM4 module has two `POWER-OUT` pins on +1.8V intentionally tied together) trip strict ERC. Default `run_erc=False` and expose an `--erc` flag.

### Verify with bit-exact diff

`diff_nets.py` loads both `.net` files and compares:
- Set of component refs (must be equal).
- For each ref: `value` and `footprint` must match.
- For each net: compare by `frozenset((ref, pin), ...)` member-set, NOT by name. KiCad renames unnamed nets and prefixes hier nets with the sheet path; member-set is the only stable identity.

Success criterion:
```
Components: orig=N new=N        — 0 missing, 0 extra, 0 drift
Nets: orig=M new=M               — missing=0 extra=0
```

---

## Step 4 — Generate KiCad 10 `.kicad_sch` files

This is the hard part because the format is undocumented and Eeschema is
strict. Below is every rule learned the hard way.

### File layout

```
<PROJECT>.kicad_pro            (JSON; register every sheet in "sheets")
<PROJECT>.kicad_sch            (root)
sheets/
   usb2_hub.kicad_sch
   rtc_wakeup_fan.kicad_sch
   ...
```

In the `.kicad_pro` JSON, the `"sheets"` array lists `[uuid, name]` pairs for
EVERY sheet including the root:

```json
"sheets": [
  ["<root-uuid>", "Root"],
  ["<usb2_hub-uuid>", "USB 2.0 Hub"],
  ["<rtc-uuid>", "RTC, Wakeup & FAN"],
  ...
]
```

UUIDs must be deterministic across runs (use `uuid.uuid5(namespace, key)`),
otherwise re-running the generator invalidates every reference.

### Skeleton of a `.kicad_sch`

```
(kicad_sch
    (version 20250610)
    (generator "your-tool-name")
    (generator_version "10.0")
    (uuid "<file-uuid>")
    (paper "A3"|"A2"|"User" 250.0 180.0)
    (title_block (title "...") (rev "1") (company "..."))
    (lib_symbols
        (symbol "CM4IO:Foo" ...)
        ...
    )
    ;; content blocks: (symbol ...), (wire ...), (label ...),
    ;; (global_label ...), (hierarchical_label ...), (sheet ...)
    (sheet_instances
        (path "/" (page "1"))
    )
    (embedded_fonts no)
)
```

### lib_symbols

A workable minimal symbol:

```
(symbol "CM4IO:Foo"
    (pin_numbers (hide no))            ; (hide yes) for 2-pin passives
    (pin_names (offset 0.508))
    (exclude_from_sim no) (in_bom yes) (on_board yes)
    (duplicate_pin_numbers_are_jumpers no)
    (property "Reference" "U" (at 0 H_above 0) (effects (font (size 1.27 1.27))))
    (property "Value" "Foo" (at 0 -H_above 0) (effects (font (size 1.27 1.27))))
    (property "Footprint" "" (at 0 0 0) (hide yes) (effects (font (size 1.27 1.27))))
    (property "Datasheet" "" (at 0 0 0) (hide yes) (effects (font (size 1.27 1.27))))
    (property "Description" "" (at 0 0 0) (hide yes) (effects (font (size 1.27 1.27))))
    (symbol "Foo_0_1"               ; body graphics
        (rectangle (start -W/2 -H/2) (end W/2 H/2)
            (stroke (width 0.254) (type default))
            (fill (type background))))
    (symbol "Foo_1_1"               ; pin definitions
        (pin passive line
            (at -W/2 - 2.54 Y 0)    ; angle 0 = pin points to +X (toward body)
            (length 2.54)
            (name "MOSI"  (effects (font (size 1.27 1.27))))
            (number "1"    (effects (font (size 1.27 1.27)))))
        ...
    )
    (embedded_fonts no)
)
```

**Pin geometry — the bug that took two iterations to spot**:

The `(at x y angle)` of a `(pin ...)` inside a lib_symbol IS the
**electrical connection point (the tip)**, NOT the body-attached end of the
pin. The graphical body of the pin extends BACKWARD from `at` by `length` in
the `angle+180°` direction.

```
left  pin: (at -W/2 - 2.54  y    0)   ; angle 0   = tip on the left
right pin: (at  W/2 + 2.54  y  180)   ; angle 180 = tip on the right
top   pin: (at  x   y_top  270)       ; angle 270 = tip on top
bot   pin: (at  x   y_bot   90)       ; angle 90  = tip on bottom
```

For a 2-pin passive (`Device:C`, `Device:R`, `LED`, etc.) the convention is:
- Pin 1 at `(0, +3.81, 270)` length 2.54 → tip at `(0, +3.81)` libedit.
- Pin 2 at `(0, -3.81,  90)` length 2.54 → tip at `(0, -3.81)` libedit.

### Y axis flip on placement

**Lib_symbols use +Y = up** (LibEdit convention).
**Schematic placement uses +Y = down** (paper convention).
KiCad implicitly flips Y when placing a symbol with rotation 0.

So a pin whose lib_symbol tip is at libedit `(rx, +ry)` ends up at schematic
coordinate `(cx + rx, cy - ry)` when the symbol is placed at `(cx, cy)`. Bake
the flip into `pin_positions()`:

```python
def pin_positions(libpart, geom):
    # ...
    if is_passive_2pin:
        out[str(pins[0]["num"])] = (0,  3.81, "top")     # libedit +Y = up
        out[str(pins[1]["num"])] = (0, -3.81, "bottom")  # libedit -Y = down
        return out
    # for IC-style: left tips at -W/2 - 2.54, right tips at +W/2 + 2.54
    # and oy stored in libedit coords (positive = up)
    # then: tip_x = cx + ox, tip_y = cy - oy  (single Y-flip)
```

### Grid snap — the OTHER bug

KiCad's default schematic grid is **50 mil = 1.27 mm**. A label whose anchor
is even 0.508 mm off the wire endpoint does NOT merge with the wire; the
exporter treats the pin as floating, every node ends up on a separate
`unconnected-(...)` net, and a netlist diff shows the design completely
unwired even though it looks fine visually.

Rules:

1. Every wire endpoint, label anchor, pin tip, and hierarchical_label
   position **must** snap to the 1.27 mm grid.
2. Body sizes must be multiples of 2.54 mm so body-center + body-half-width
   stays on grid: `body_w = round(want_w / 2.54) * 2.54`.
3. The label `(at x y rot)` anchor is the same `(x, y)` as the wire endpoint
   — DO NOT add a visual padding offset. Place the label at the wire's
   endpoint exactly.

```python
def snap(v): return round(v / 1.27) * 1.27
```

Snap everything coming out of the placement pass, including the stub
endpoints.

### Wires and labels — prefer label-at-tip over stub wires

The simplest and most reliable strategy is to place a label directly at each
pin tip coordinate. KiCad merges a label with any pin whose electrical tip
is at the same (x, y). No wires are needed at all. This avoids an entire
class of bugs (see §Debugging cookbook #9 — collinear wire overlap).

If you do choose to use stub wires for visual clarity, stub length 5.08 mm
(4 grid units) works. But beware: **never let stub wires from different
components be collinear on the same row or column**, as KiCad will merge
them into one net. The safe default is label-at-tip.

Direction for stub wires (if used):
- side=left  →  `dx = -5.08`
- side=right →  `dx = +5.08`
- side=top   →  `dy = -5.08`   (schematic +Y is down, so "top" is -Y)
- side=bottom→  `dy = +5.08`

### Labels — which kind?

| Net class | Label type | Effect |
|-----------|------------|--------|
| local     | `(label ...)` | merges with same-named labels on the **same sheet** |
| hier      | `(global_label ...)` | merges across **all sheets** in the project |
| power     | `(global_label ...)` (and/or `power:GND`/`power:+5V` port symbols) | same |

**Hierarchical_labels alone do NOT carry connectivity across the
project.** They only define what the parent sheet's sheet pins should be.
Therefore, for every cross-sheet net you need BOTH:

- a `global_label` at every component pin (for actual electrical merging), AND
- ONE `hierarchical_label` somewhere on the child sheet (so the parent's
  sheet-pin auto-population picks it up).

Co-locate the lone `hierarchical_label` with a `global_label` of the same
name so it isn't orphan-floating.

### Label anchor rotation

The label `(at x y rot)` rotation determines text justification direction:

| side  | rot | justify |
|-------|----:|---------|
| left  | 180 | right   |
| right |   0 | left    |
| top   |  90 | left    |
| bot   | 270 | right   |

Anchor `(x, y)` stays exactly on the wire endpoint.

### Power port symbols

When power is shown as a port symbol rather than a global label, include this
in `lib_symbols`:

```
(symbol "power:GND" (power) ...)
(symbol "power:+5V" (power) ...)
```

A power symbol is just a normal `(symbol ...)` instance with `lib_id
"power:GND"` etc.; the (power) flag in the lib symbol makes it a global net
sink/source.

### Hierarchical sheets on the root

```
(sheet
    (at <x> <y>)
    (size <w> <h>)
    (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)
    (fields_autoplaced yes)
    (stroke (width 0.1524) (type solid))
    (fill (color 0 0 0 0.0000))
    (uuid "<sheet-uuid>")
    (property "Sheetname" "USB 2.0 Hub" (at <x> <y-0.7> 0)
        (effects (font (size 2.0 2.0) (thickness 0.35) (bold yes))
                 (justify left bottom)))
    (property "Sheetfile" "sheets/usb2_hub.kicad_sch" (at <x> <y+h+1.3> 0)
        (hide yes)
        (effects (font (size 1.27 1.27)) (justify left top)))
    ;; One (pin ...) per cross-sheet net (matches hier_label on the child)
    (pin "SCL0" input
        (at <x> <pin_y> 180)
        (uuid "<deterministic>")
        (effects (font (size 1.27 1.27)) (justify right)))
    ;; ...
    (instances
        (project "<PROJECT>"
            (path "/<root-uuid>" (page "2"))))   ; page numbers start at 2
)
```

Sheet pins on the parent and hierarchical_labels on the child must MATCH BY
NAME exactly. If a parent has a pin "VBUS_EN" but the child has no
`(hierarchical_label "VBUS_EN" ...)`, ERC flags "Pin not connected".

### Symbol instances on a sheet

```
(symbol
    (lib_id "CM4IO:Foo")
    (at <x> <y> 0)
    (unit 1) (exclude_from_sim no) (in_bom yes) (on_board yes) (dnp no)
    (uuid "<deterministic>")
    (property "Reference" "U6" (at <x> <y - h/2 - 2.54> 0)
        (effects (font (size 1.27 1.27))))
    (property "Value" "USB2514B-I/M2" (at <x> <y + h/2 + 2.54> 0)
        (effects (font (size 1.27 1.27))))
    (property "Footprint" "Package_DFN_QFN:QFN-36-1EP_6x6mm_..."
        (at <x> <y> 0) (hide yes)
        (effects (font (size 1.27 1.27))))
    (property "Datasheet" "" (at <x> <y> 0) (hide yes)
        (effects (font (size 1.27 1.27))))
    (pin "1" (uuid "<deterministic-pin-uuid>"))
    (pin "2" (uuid "<...>"))
    ;; ...
    (instances
        (project "<PROJECT>"
            (path "/<root-uuid>/<sheet-uuid>"
                (reference "U6") (unit 1))))
)
```

The `(instances)` `(path ...)` must include EVERY UUID from root down to the
sheet containing the component, separated by `/`. Mounting holes on the root
itself have `path "/<root-uuid>"` — one segment only.

### Page size — autosize each sheet

Don't pick a fixed paper. Compute the content bounding box first, then choose
the smallest standard ISO size that fits, falling back to `(paper "User" W H)`
only when nothing standard works.

```python
STANDARD_PAPERS = [
    # (name, long mm, short mm) — landscape orientation
    ("A5",  210.0,  148.0),
    ("A4",  297.0,  210.0),
    ("A3",  420.0,  297.0),
    ("A2",  594.0,  420.0),
    ("A1",  841.0,  594.0),
    ("A0", 1189.0,  841.0),
]

def pick_paper(content_w, content_h, margin=15.0):
    w = content_w + 2 * margin
    h = content_h + 2 * margin
    for name, pw, ph in STANDARD_PAPERS:
        if w <= pw and h <= ph: return f'"{name}"', pw, ph
        if w <= ph and h <= pw: return f'"{name}" portrait', ph, pw
    uw = math.ceil(w / 10) * 10
    uh = math.ceil(h / 10) * 10
    return f'"User" {uw:.4f} {uh:.4f}', uw, uh
```

Bounding-box pass per sheet:

1. Place every component → bbox += component body rectangle + slack for
   ref/value text.
2. Add a margin per component side for the longest label name that will
   appear (use `max_label_w = max(0.85 * max_name_len * 1.5, 10.0)`).
   Do **NOT** iterate over all label names adding text width each time —
   that double-counts and inflates the bbox by O(N_labels).
3. For every hierarchical sheet-pin label on the child's left edge →
   bbox += that label too.
4. Pick paper. Compute `pad_x = (paper_w − content_w) / 2`, same for y.
5. Shift = `pad - bbox_min`. Re-emit every coordinate through
   `sx(v) = snap(v + shift_x)`.

Result on a real board: from a fixed A1 for every page (huge whitespace) to
A3 / A2 / A1 chosen per sheet, content centered, no glitching out of bounds.

### Component placement — topology-aware

The netlist contains all the information needed for coherent, hand-quality
placement. No vision model or external reference needed — extract "design
intent" from connectivity patterns.

#### Step 1: Classify each small component by role

Before placing anything, walk every component with ≤2 pins and determine its
role from the nets it touches:

| Role | Topological signature | Placement rule |
|------|-----------------------|----------------|
| **Decoupling cap** | Pin 1 on a power net (`+3.3v`, `+5v`, etc.), pin 2 on `GND` | Place adjacent to the IC pin that shares the same power net, offset in the pin's outward direction |
| **Pull-up** | One pin on a power net, other pin on a signal net | Place near the signal net's other endpoint (usually an IC pin) |
| **Pull-down** | One pin on `GND`, other pin on a signal net | Same as pull-up |
| **Crystal load cap** | One pin on `XTALIN` or `XTALOUT`, other pin on `GND` | Group with crystal, near the XTAL pin of the IC |
| **Series resistor** | Both pins on signal nets (neither power nor GND) | Place between the two endpoints of its nets |
| **LED resistor** | One pin on a signal net, other on an LED or power | Place near the LED |
| **Filter cap** | One pin on a named local net (e.g. `Net-(U15-FB)`), other on `GND` | Place near the IC pin that owns that named net |
| **Bulk/energy cap** | Pin 1 on power, pin 2 on GND, but large value (≥47µF) | Place near the power source or regulator, not per-pin |

Algorithm for classification:

```python
def classify(ref, comp, nets_by_pin):
    """Return (role, anchor_ref, anchor_pin) for a 2-pin component."""
    net_a, net_b = nets_by_pin[ref]  # {pin_num: net_name}
    is_power_a = is_power_net(net_a)
    is_gnd_a = (net_a == "GND")
    is_power_b = is_power_net(net_b)
    is_gnd_b = (net_b == "GND")

    if is_power_a and is_gnd_b:
        return ("decoupling", find_ic_sharing_net(ref, net_a, "1"), net_a)
    if is_gnd_a and is_power_b:
        return ("decoupling", find_ic_sharing_net(ref, net_b, "2"), net_b)
    if is_power_a and not is_gnd_b:
        return ("pull-up", find_ic_sharing_net(ref, net_b, "2"), net_b)
    if is_gnd_a and not is_power_b:
        return ("pull-down", find_ic_sharing_net(ref, net_b, "2"), net_b)
    if not is_power_a and not is_gnd_a and not is_power_b and not is_gnd_b:
        return ("series", None, None)
    # Named-net filter cap: one pin is "Net-(Uxx-PIN)", other is GND
    if is_gnd_a and is_named_net(net_b):
        return ("filter", find_ic_from_net_name(net_b), net_b)
    if is_gnd_b and is_named_net(net_a):
        return ("filter", find_ic_from_net_name(net_a), net_a)
    return ("generic", None, None)
```

`find_ic_sharing_net(small_ref, net_name, small_pin)` walks the net's node
list, skips `small_ref` itself, and picks the first IC-class component (>
2 pins). If multiple ICs share the net (e.g. a power rail), pick the one
with the highest pin count — it's the most likely "anchor".

#### Step 2: Place anchor components first

Anchor components are ICs (ref prefix U, Module, Y) and connectors (J).
Place them in a logical signal-flow order:

1. **Identify signal flow**: Build a directed graph of hier/signal nets
   (not power/GND). Components with only outputs on the left; only inputs
   on the right; mixed in the middle.
2. **Place left-to-right, top-to-bottom**: Main signal path flows left → right.
   Within a column, place top → bottom.
3. **Stack large ICs vertically**: For components >8 pins, give each its own
   row. Body center Y = first_row + row_index × row_pitch. Row pitch =
   max(body_h) + 40mm.
4. **Connectors at edges**: Place input connectors on the left margin,
   output connectors on the right margin.

#### Step 3: Place satellite components near their anchors

Walk each small component in classification order:

- **Decoupling caps**: Place offset from the anchor pin in the pin's outward
  direction by (5.08mm perpendicular, 2.54mm parallel). If the pin is on
  the IC's right side, the cap goes slightly further right and vertically
  aligned with the pin. Stack multiple decoupling caps for the same IC
  vertically, 5.08mm apart.
- **Pull-up / pull-down resistors**: Place on the same side as the signal
  pin, offset outward by 10.16mm, aligned to the pin's Y coordinate.
- **Crystal + load caps**: Group together. Place the crystal below the IC
  (or above, depending on XTAL pin location). Load caps go between crystal
  and GND, stacked vertically beside the crystal.
- **Series resistors**: Place midway between their two net endpoints.
- **Filter caps**: Place adjacent to the named-net IC pin.

#### Step 4: Assign pin sides per component

For each IC, group its pins by functional role before assigning sides:

```python
def assign_pin_sides(ic_ref, libpart, nets_by_pin):
    """Decide which pins go on which side of the symbol body."""
    groups = defaultdict(list)
    for pin in libpart["pins"]:
        net = nets_by_pin.get(pin["num"], "")
        if is_power_net(net) or net == "GND":
            groups["power"].append(pin)
        elif net.startswith("unconnected-"):
            groups["nc"].append(pin)
        elif pin["type"] in ("input", "clock"):
            groups["input"].append(pin)
        elif pin["type"] in ("output", "tri_state"):
            groups["output"].append(pin)
        elif pin["type"] == "bidirectional":
            groups["io"].append(pin)
        else:
            groups["passive"].append(pin)

    # Convention: inputs left, outputs right, power top, GND bottom
    sides = {"input": "left", "output": "right",
             "io": "right", "power": "top",
             "passive": "left", "nc": "bottom"}
    return {pin["num"]: sides[grp] for grp, pins in groups.items()
            for pin in pins}
```

This replaces the naive `i % 4` round-robin and produces schematics where
signal flow is visible: inputs arrive from the left, outputs leave to the
right, power is at the top, and ground is at the bottom.

#### Step 5: Fine-tune with spacing rules

- Minimum gap between component bodies: 5.08mm (4 grid units)
- Decoupling cap to IC pin: 2.54mm offset
- Label stub length: 5.08mm (enough for label text to clear the pin wire)
- Power rail labels (GND, +3.3v, +5v, +12v): place at the far end of each
  stub, oriented so text reads naturally (GND at bottom pointing down,
  +V at top pointing up)
- No two labels should share the same (x, y) — if they would, nudge one
  by 2.54mm in the pin's direction

#### Fallback: if classification is ambiguous

If a small component doesn't match any role (e.g. both pins on unnamed
local nets), fall back to the simple grid placement from the original
algorithm. This keeps output valid even if the topology analysis can't
classify everything.

#### Why this works without a vision model

The netlist IS the schematic topology. Every "design intent" pattern
(decoupling, pull-up, crystal, series termination) leaves a unique
topological fingerprint in the connectivity graph. A vision model reading
a reference schematic can only tell you what the netlist already says —
plus it hallucinates. Algorithmic extraction is deterministic, verifiable,
and requires no external model call.

### Deterministic UUIDs

Re-running the generator must not invalidate references. Use a fixed namespace:

```python
_NS = uuid.UUID("12345678-1234-1234-1234-1234567890ab")
def det_uuid(*parts): return str(uuid.uuid5(_NS, "|".join(map(str, parts))))
```

Key UUIDs by content: `det_uuid("sym", ref)`, `det_uuid("pin", ref, pinnum)`,
`det_uuid("wire", ref, pin, netcode)`, etc.

### Semantic page descriptions

Every generated schematic sheet gets a **description text block** near the
top-left corner, just below the title block. This provides at-a-glance
context for anyone opening the sheet — what the page does, which ICs are
central, how power flows, and what the key signal paths are.

#### Content structure

Each sheet description follows this template:

1. **Title line** (bold, 3mm font): Sheet name and functional role.
2. **Central IC** (if any): Part number, reference, and what it does.
3. **Signal flow**: How signals enter and leave the sheet.
4. **Power architecture**: Which rails are used, how they're generated.
5. **Key sub-circuits**: Decoupling, pull-ups, crystal, ESD, etc.
6. **Connector roles**: What each connector connects to.

#### Implementation

Add a `SHEET_DESCRIPTIONS` dict keyed by sheet path (`"/"`, `"/USB2-HUB/"`,
etc.). Each value is a list of strings — one per line. The first line is
the bold title; subsequent lines are body text at 2mm font.

```python
SHEET_DESCRIPTIONS = {
    "/USB2-HUB/": [
        "USB 2.0 Hub - 4-Port Expansion",
        "Central IC: Microchip USB2514B (U6) - 4-port USB 2.0 hub controller.",
        "Expands the single upstream USB 2.0 port from the CM4 into four",
        "downstream ports (J11, J13, J14). Two ports are protected by",
        "current-limiting power switches (AP2553W6 - U4, U7). A 24 MHz",
        "crystal (Y1) provides the reference clock. Decoupling capacitors",
        "(C2-C21) filter power supply noise across +3.3V, +5V, and GND.",
    ],
    ...
}
```

Emit as individual `(text ...)` S-expression elements, **not** a
`(text_box ...)`. KiCad 10's `text_box` format has strict syntax that
varies between versions; simple `(text ...)` blocks are portable and
reliable.

```python
def gen_description_text(sname, start_x, start_y, line_spacing=3.81):
    lines = []
    desc_lines = SHEET_DESCRIPTIONS.get(sname, [])
    if not desc_lines:
        return lines, 0, 0
    y = start_y
    for i, dl in enumerate(desc_lines):
        txt_uuid = det_uuid("desc", sname, i)
        font_size = 3.0 if i == 0 else 2.0
        bold = " bold" if i == 0 else ""
        lines.append(f'  (text "{escape_str(dl)}" (at {snap(start_x)} {snap(y)} 0)')
        lines.append(f'    (uuid "{txt_uuid}")')
        lines.append(f'    (effects (font (size {font_size} {font_size}){bold}) (justify left top)))')
        y += line_spacing
    return lines, max_w, len(desc_lines) * line_spacing
```

Insert the text blocks into the `.kicad_sch` after `(lib_symbols ...)`
and before component `(symbol ...)` instances.

#### How to write good descriptions

- **Infer from the netlist**: The component list, net names, and pin
  types are enough. E.g. a sheet with `USB2514B` + 4 `USB-A` connectors
  + `AP2553W6` power switches is clearly a USB hub with per-port
  overcurrent protection.
- **Name specific reference designators**: "U6" not "the hub IC",
  "C3-C7" not "decoupling caps". This cross-references the schematic.
- **Mention all power rails**: "+3.3V, +5V, and GND" — tells the reader
  which supply domains are active on this page.
- **Keep it under 10 lines**: The text block should fit above the first
  component row without pushing content off-page.
- **Bold title line only**: The first line is 3mm bold; body is 2mm
  regular. This gives a clear visual hierarchy.

---

## Step 5 — Verify via round-trip

Two verifications, both done by `kicad-cli` so the result is exactly what
KiCad would do at load time:

```bash
KICAD="/mnt/c/Program Files/KiCad/10.0/bin/kicad-cli.exe"   # Windows + WSL path

# Re-export a netlist from our generated .kicad_sch
"$KICAD" sch export netlist --format kicadsexpr \
    -o <PROJECT>.from_sch.net <PROJECT>.kicad_sch

# Diff it against the original netlist (member-set comparison; see Step 3)
python3 diff_nets.py     # must report: missing=0 extra=0 drift=0

# Visual smoke test
"$KICAD" sch export pdf -o <PROJECT>.pdf <PROJECT>.kicad_sch
```

If `from_sch.net` doesn't match the original, the diff will show specific
nets and (ref, pin) pairs that drifted — use those to trace which label or
wire is off.

Optional but useful: `"$KICAD" sch erc --severity-all` to surface remaining
floating pins / orphan signals.

---

## Debugging cookbook

The five recurring failure modes from this work:

### 1. Doubled components in the SKiDL netlist (`Module1_1`, `C1_1`, ...)

Cause: generator emitted two `if __name__ == "__main__":` blocks; or some
caller does `import package; package.main()` after `python -m package`
already ran. Verify: `grep -c '^if __name__' <top>.py` should print `1`.

### 2. Every pin on its own `unconnected-(...)` net

Cause #1: label is off the 1.27 mm grid (e.g. you added `±0.508 mm` for
"label padding"). Snap all coords to 1.27 mm.

Cause #2: label is on a different sheet than the matching name uses (used a
local `(label ...)` instead of `(global_label ...)` for a cross-sheet net).

Cause #3: matching labels disagree on text (`/CM4_HighSpeed/SCL0` on one
sheet, `SCL0` on another). Display name must be the leaf (`SCL0`) everywhere.

### 3. Components placed but wires go to nowhere visible

Cause: confusing `at` with body-edge in lib_symbols. The `at` IS the tip; do
not add `length` to it. The pin body extends in the `angle+180°` direction
INTO the component body.

### 4. Pin positions are mirrored top-to-bottom

Cause: forgot the Y-axis flip between libedit (+Y up) and schematic (+Y
down). Add the flip in `pin_positions()`.

### 5. Parent sheet shows bare rectangles instead of sheet pins

Cause: child sheet only has `global_label`s, no `hierarchical_label`s. Add
ONE `(hierarchical_label ...)` per cross-sheet net on the child (and a
co-located `(global_label ...)` so the hier label has connectivity), then
emit matching `(pin "NAME" ...)` entries on the parent's `(sheet ...)`
block.

### 6. ERC complains about POWER-OUT pin conflicts

Cause: the design itself ties two POWER-OUT pins of one IC to the same rail
(common on compute modules with `+1.8V_(Output)` on two pins). Hardware is
correct; ERC is over-strict. Leave the pin types as-is from the netlist;
just default-off ERC in your generator, expose `--erc` for opt-in.

### 7. Eeschema opens but a sheet is blank / KiCad complains "missing
   library symbol"

Cause: a `(symbol ...)` instance has `lib_id "CM4IO:Foo"` but the file's
`lib_symbols` block has no `(symbol "CM4IO:Foo" ...)`. Each sheet must
include lib_symbols for EVERY part placed on it (root included, e.g. for
mounting holes on the root).

### 8. Pin graphic type "power" is rejected by KiCad 10

Cause: using `power` as the electrical type in a `(pin ...)` definition
inside `(lib_symbols ...)`. KiCad 10 only accepts the 12 standard electrical
types: `input`, `output`, `bidirectional`, `tri_state`, `passive`, `free`,
`unspecified`, `power_in`, `power_out`, `open_collector`, `open_emitter`,
`no_connect`. The type `power` does **not** exist as a pin graphic type,
even though KiCad internally renders power pins differently. If you map
`power_in`/`power_out` → `"power"` in your generator, KiCad will silently
fail to load the sheet (zero components in the exported netlist, or "Failed
to load schematic" with exit code 3). Fix: pass the original
`power_in`/`power_out` type through unchanged.

```python
def pin_type_to_graphic(ptype):
    valid = {"input", "output", "bidirectional", "tri_state", "passive",
             "free", "unspecified", "power_in", "power_out",
             "open_collector", "open_emitter", "no_connect"}
    if ptype in valid:
        return ptype
    return "passive"  # safe fallback
```

### 9. Collinear stub wires overlap and silently merge nets

Cause: when two components are placed on the same horizontal or vertical
line, their stub wires (e.g. 5.08 mm extending left from each pin tip) can
be collinear and overlapping. KiCad treats overlapping collinear wire
segments as a single connected wire, so a label intended for one component's
pin electrically connects to the adjacent component's pin as well. This
causes massive net-merge contamination (e.g. GND absorbing dozens of signal
nets).

Example: Component A pin 1 at x=100, stub to x=94.92; Component B pin 1 at
x=94.92, stub to x=89.84. These overlap on [89.84, 94.92] and merge.

Fix: **do not use stub wires at all**. Place labels directly at pin tip
positions. KiCad connects a label to any pin whose tip is at the same
coordinate as the label anchor. This eliminates the overlap class of bugs
entirely and produces bit-exact netlists.

### 10. kicad-cli cannot load a child sheet standalone

Cause: `kicad-cli.exe sch export netlist sheets/foo.kicad_sch` fails with
"Failed to load schematic" (exit code 3). Child sheets are only valid in
the context of their parent project. You must always export from the root:
`kicad-cli.exe sch export netlist project.kicad_sch`.

### 11. Y-flip double-negation for 2-pin passives in pin_positions()

Cause: the skill already covers the Y-flip (`cy - oy`), but a subtle trap
is storing offsets in *schematic* coordinates in `pin_positions()`. For a
2-pin vertical passive, pin 1 is at libedit y=+3.81 (top in libedit). If
you store `oy = -3.81` in pin_positions (thinking "top in schematic = -Y"),
then `cy - (-3.81) = cy + 3.81` — which is *bottom* in the schematic. This
double-flips. The correct approach: store offsets in **libedit coordinates**
(`oy = +3.81` for pin 1), so that `cy - (+3.81) = cy - 3.81` correctly maps
to the *top* of the component in schematic coordinates.

```python
if is_passive and n_pins == 2:
    return {
        str(pins[0]['num']): (0,  3.81, 'top'),     # libedit +Y = up
        str(pins[1]['num']): (0, -3.81, 'bottom'),  # libedit -Y = down
    }
# Then: tip_y = cy - oy  (single flip to schematic coordinates)
```

### 12. Bounding-box loop adds phantom padding per label name

Cause: a loop in `_compute_bbox` that iterates over every label name and
adds `text_width` to both `min_x` and `max_x` (or `min_y`/`max_y`). With
100+ net names on a typical sheet, this inflates the computed content
size by thousands of millimeters, guaranteeing that `pick_paper` falls
through every standard ISO size and emits a `"User"` page — often
absurdly skinny (e.g. 5760 × 300 mm for CM4_HighSpeed).

The per-component margin (adding `max_label_w` once per component side)
already accounts for label text width. Adding it again per label name is
a double-count that scales with `O(N_labels)` and dominates the bbox on
real boards.

Fix: remove the per-label-name padding loop entirely. Use a single
`max_label_w` (computed from the longest label name) added once per
component side, or use the body rectangle + a fixed margin.

### 13. Large ICs placed in a single horizontal row produce unusable pages

Cause: the placement algorithm puts all "large" components (>8 pins)
side-by-side in one row. A 200-pin component with 71 mm pin offsets per
side occupies ~280 mm of horizontal space by itself. Putting 6
connectors alongside it produces a content width of 5000+ mm, which no
standard paper can fit.

Fix: stack large components **vertically** (one per row, centered on the
page width). This produces a portrait-oriented layout that fits A1/A2
portrait naturally. Small components go in a multi-column grid below the
large ones.

```python
# Place large components stacked vertically, centered
y = 50.0
for ref in large:
    cx = snap(50.0 + body_w / 2)   # center on page
    placements[ref] = (cx, snap(y + body_h / 2), body_w, body_h, pin_pos)
    y += body_h + 40.0
```

### 14. Label rotation left/right swapped — labels point into the body

Cause: the `_label_rot_justify` function had left and right swapped
relative to the documented table. Left-side pins got `rot=0,
justify=left` (text baseline runs left→right, so text extends **right**
from the anchor — straight into the component body). Right-side pins got
`rot=180, justify=right` (text extends **left** from the anchor — also
inward). This makes many labels unreadable because they overlap the
symbol body.

The correct mapping (matching the skill's existing table) is:

| side  | rot | justify | text direction from anchor |
|-------|----:|---------|----------------------------|
| left  | 180 | right   | rightward ← anchor (outward) |
| right |   0 | left    | leftward ← anchor (outward) |
| top   |  90 | left    | downward ← anchor (outward) |
| bot   | 270 | right   | upward ← anchor (outward)   |

A quick way to verify: for a left-side pin, the label anchor sits at
the pin tip (left of body). Text at `rot=180` with `justify=right` means
the text reads right-to-left starting from the anchor, so it extends
**left** — away from the body. If you see labels crawling under the
rectangle, swap left↔right in your rotation map.

### 15. Labels appear visually disconnected from pins on multi-pin ICs

Cause: `pin_positions()` and the `lib_symbol` pin `(at x y angle)` must
agree on the Y coordinate. The lib_symbol uses **libedit Y-up** convention,
while the schematic renders with **Y-down**. KiCad implicitly flips Y when
placing a symbol at rotation 0: a pin at libedit `(rx, +ry)` renders at
schematic `(cx + rx, cy - ry)`.

If your `pin_positions()` stores offsets in libedit coordinates (oy
positive = up) and your label placement uses `tip_y = cy - oy`, then the
label anchor lands at `cy - oy` — exactly where KiCad renders the pin tip
after its internal Y-flip. This is correct.

The trap is accidentally negating Y **twice**: once in `pin_positions()`
(storing oy in schematic Y-down) and again in the label placement
(`cy - oy`). This double-flip puts the label on the **opposite** side of
the body from the pin. The round-trip netlist diff catches the resulting
pin swaps (e.g. `U6.10` connects to `GPIO15` label that was meant for
`U6.15`), but it looks fine visually until you inspect individual nets.

Rule: **store pin offsets in libedit Y-up coordinates. Apply the single
flip `cy - oy` only at label-placement time. Never negate Y in the
lib_symbol `(at x y angle)` — that coordinate IS libedit space.**

### 16. `(text_box ...)` fails to load in KiCad 10

Cause: KiCad 10's `text_box` S-expression format has strict syntax
requirements (stroke width, fill type, size fields) that are easy to
mis-format. A misplaced field or wrong fill type silently causes
"Failed to load schematic" with no parse error details.

Fix: use simple `(text ...)` blocks instead. Each line of the
description is a separate `(text ...)` element with a UUID and
`(effects (font (size ...) ...) (justify left top))`. These are
portable across all KiCad versions and always parse correctly.

The `bold` keyword must be a sibling to `(size ...)` inside `(font ...)`,
NOT inside `(size ...)`:

```python
# CORRECT: bold is outside size
'(effects (font (size 3.0 3.0) bold) (justify left top))'

# WRONG: bold is inside size — KiCad rejects this
'(effects (font (size 3.0 3.0 bold)) (justify left top))'
```

---

## Tools and environment

- **Python 3.10+** with `skidl` installed (pip).
- On WSL pointing at Windows-installed KiCad:
  `/mnt/c/Program Files/KiCad/10.0/bin/kicad-cli.exe`
  `/mnt/c/Program Files/KiCad/10.0/bin/eeschema.exe`
  Pass Windows paths to these binaries (`C:\\Users\\foo\\...`).
- `kicad-cli` subcommands you'll actually use:
  - `sch erc --severity-all --format report -o report.txt input.kicad_sch`
  - `sch export netlist --format kicadsexpr -o out.net input.kicad_sch`
  - `sch export pdf -o out.pdf input.kicad_sch`

KiCad 10 ships demo schematics under
`/mnt/c/Program Files/KiCad/10.0/share/kicad/demos/`. The **cm5_minima**
demo is an excellent format reference — it is a hierarchical Raspberry Pi
CM5 IO board very similar to what this skill targets.

---

## Recommended Q&A flow

Before generating, ask:

1. **Output target** — Python SKiDL + netlist only, or also `.kicad_sch`
   files? (Default to *also* schematic if the user said "recreate the
   schematic".)
2. **Library symbol strategy** — synthesise custom TEMPLATE Parts from
   `libparts` (always works, reproduces design exactly), or try to map to
   stock KiCad libs (`Device:R`, `Timer_RTC:PCF8563T`, etc.) where possible
   and fall back to TEMPLATE for custom ones. Default to TEMPLATE if unsure.
3. **Top-level layout** — bare sheet rectangles (clean) or sheet pins per
   block (verbose but proper hierarchical). Default to **sheet pins** —
   without them, page 1 is just empty boxes.

After generating, always:

- Run the netlist round-trip diff and report the numbers
  (`151/151 components, 247/247 nets, 0 drift`).
- Render a PDF and report its size as a smoke test.
- Offer to launch Eeschema on the result.

---

## Anti-patterns to avoid

- **Package named `skidl`** — shadows the library.
- **Adding padding offset to label anchors** — breaks net merging.
- **Random UUIDs per run** — invalidates instance references.
- **Fixed paper size for every sheet** — wastes A1 on a 3-component sheet or
  truncates a 200-pin part on A4.
- **Sheet pins on parent without hierarchical_labels on child** — ERC
  errors, "Pin not connected".
- **Hierarchical_labels alone for cross-sheet connectivity** — they don't
  actually merge nets project-wide; use global_labels too.
- **Trusting net names** instead of (ref, pin) member-sets for diffs —
  KiCad renames freely.
- **Calling `main()` twice** anywhere — SKiDL is module-stateful, you get a
  duplicated netlist.
- **Pin graphic type `"power"`** — not valid in KiCad 10; use `power_in` or
  `power_out` directly.
- **Stub wires from collinear pins** — overlapping collinear wire segments
  silently merge into a single net; prefer label-at-tip with no wires.
- **Running kicad-cli on a child sheet** — always export from the root
  `.kicad_sch`; child sheets are not loadable standalone.
- **Per-label-name bbox padding loop** — adding text-width padding once
  per net name double-counts and inflates the bbox by O(N_labels); use
  a single `max_label_w` per component side instead.
- **Large ICs in a single horizontal row** — a 200-pin part with 71 mm
  pin offsets per side makes the row absurdly wide; stack large
  components vertically (portrait layout) instead.
- **Left/right label rotation swapped** — left-side pins must use
  `rot=180, justify=right` (text extends left, away from body), not
  `rot=0, justify=left` (text extends right, into body). If labels
  overlap the symbol rectangle, your rotation map is inverted.
- **Using `text_box` for sheet descriptions** — KiCad 10 `text_box`
  syntax is fragile and varies between sub-versions; use multiple
  `(text ...)` blocks instead, one per line.
- **`bold` inside `(size ...)`** — must be `(font (size 3 3) bold)`,
  not `(font (size 3 3 bold))`; the latter silently fails to parse.
- **Satellite placement near connectors** — connectors (J prefix) have
  many pins with different nets packed tightly; placing decoupling caps
  near them causes label collisions and net merges. Only place
  satellites near ICs (U/M/Y prefix).

---

## Success criteria

A successful application of this skill produces:

1. A `<PROJECT>_skidl/` Python package whose `python -m <PROJECT>_skidl.<top>`
   prints `INFO: 0 errors found while generating netlist` and writes a
   `<PROJECT>.skidl.net` that matches the original 1:1.
2. A `<PROJECT>.kicad_sch` + `sheets/*.kicad_sch` set that opens in Eeschema
   10 with zero parse errors, page sizes fitted to content, and a top-level
   page showing real hierarchical blocks with sheet pins.
3. `kicad-cli sch export netlist` on the generated schematic produces a
   netlist whose member-set diff against the original reports
   `missing=0 extra=0 drift=0`.

If any one of the three fails, return to the debugging cookbook above
before declaring done.
