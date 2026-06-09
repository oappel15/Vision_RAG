from collections import defaultdict
from skidl import Pin, Part, Alias, SchLib, SKIDL, TEMPLATE

from skidl.pin import pin_types

SKIDL_lib_version = '0.0.1'

attiny85_blinker_skidl = SchLib(tool=SKIDL).add_parts(*[
        Part(**{ 'name':'ATtiny85-20P', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'ATtiny85-20P'}), 'ref_prefix':'U', 'fplist':['Package_DIP:DIP-8_W7.62mm', 'Package_DIP:DIP-8_W7.62mm'], 'footprint':'Package_DIP:DIP-8_W7.62mm', 'keywords':'AVR 8bit Microcontroller tinyAVR', 'description':'20MHz, 8kB Flash, 512B SRAM, 512B EEPROM, debugWIRE, DIP-8', 'datasheet':'http://ww1.microchip.com/downloads/en/DeviceDoc/atmel-2586-avr-8-bit-microcontroller-attiny25-attiny45-attiny85_datasheet.pdf', 'pins':[
            Pin(num='1',name='~{RESET}/PB5',func=pin_types.BIDIR,unit=1),
            Pin(num='2',name='XTAL1/PB3',func=pin_types.BIDIR,unit=1),
            Pin(num='3',name='XTAL2/PB4',func=pin_types.BIDIR,unit=1),
            Pin(num='4',name='GND',func=pin_types.PWRIN,unit=1),
            Pin(num='5',name='AREF/PB0',func=pin_types.BIDIR,unit=1),
            Pin(num='6',name='PB1',func=pin_types.BIDIR,unit=1),
            Pin(num='7',name='PB2',func=pin_types.BIDIR,unit=1),
            Pin(num='8',name='VCC',func=pin_types.PWRIN,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'R', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'R'}), 'ref_prefix':'R', 'fplist':[''], 'footprint':'Resistor_THT:R_Axial_DIN0207_L6.3mm_D2.5mm_P10.16mm_Horizontal', 'keywords':'R res resistor', 'description':'Resistor', 'datasheet':'', 'pins':[
            Pin(num='1',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'C', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'C'}), 'ref_prefix':'C', 'fplist':[''], 'footprint':'Capacitor_THT:C_Disc_D5.0mm_W2.5mm_P5.00mm', 'keywords':'cap capacitor', 'description':'Unpolarized capacitor', 'datasheet':'', 'pins':[
            Pin(num='1',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'LED', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'LED'}), 'ref_prefix':'D', 'fplist':[''], 'footprint':'LED_THT:LED_D5.0mm', 'keywords':'LED diode', 'description':'Light emitting diode', 'datasheet':'', 'pins':[
            Pin(num='1',name='K',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='A',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'Conn_01x02', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Conn_01x02'}), 'ref_prefix':'J', 'fplist':[''], 'footprint':'Connector_PinHeader_2.54mm:PinHeader_1x02_P2.54mm_Vertical', 'keywords':'connector', 'description':'Generic connector, single row, 01x02, script generated (kicad-library-utils/schlib/autogen/connector/)', 'datasheet':'', 'pins':[
            Pin(num='1',name='Pin_1',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='Pin_2',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] }),
        Part(**{ 'name':'Conn_02x03_Odd_Even', 'dest':TEMPLATE, 'tool':SKIDL, 'aliases':Alias({'Conn_02x03_Odd_Even'}), 'ref_prefix':'J', 'fplist':[''], 'footprint':'Connector_PinHeader_2.54mm:PinHeader_2x03_P2.54mm_Vertical', 'keywords':'connector', 'description':'Generic connector, double row, 02x03, odd/even pin numbering scheme (row 1 odd numbers, row 2 even numbers), script generated (kicad-library-utils/schlib/autogen/connector/)', 'datasheet':'', 'pins':[
            Pin(num='1',name='Pin_1',func=pin_types.PASSIVE,unit=1),
            Pin(num='2',name='Pin_2',func=pin_types.PASSIVE,unit=1),
            Pin(num='3',name='Pin_3',func=pin_types.PASSIVE,unit=1),
            Pin(num='4',name='Pin_4',func=pin_types.PASSIVE,unit=1),
            Pin(num='5',name='Pin_5',func=pin_types.PASSIVE,unit=1),
            Pin(num='6',name='Pin_6',func=pin_types.PASSIVE,unit=1)], 'unit_defs':[] })])