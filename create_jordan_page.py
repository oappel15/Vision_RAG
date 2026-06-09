import os
import requests
import json

PAT = os.environ["CONFLUENCE_PAT"]
BASE = "http://localhost:8090"
HEADERS = {
    "Authorization": f"Bearer {PAT}",
    "Content-Type": "application/json",
}
SPACE_KEY = "RAG"
PARENT_ID = "131148"

PAGE_BODY = """
<h1>Michael Jordan — Career Overview</h1>
<p><strong>Michael Jeffrey Jordan</strong> (born February 17, 1963) is widely regarded as the greatest basketball player of all time. Over a 15-year NBA career, he won six championships, six Finals MVP awards, and five regular-season MVP awards, while revolutionizing the game with his athleticism, competitive drive, and global appeal.</p>

<h2>1. Career Statistics Summary</h2>
<table>
  <tbody>
    <tr><th>Season Type</th><th>Games</th><th>PPG</th><th>RPG</th><th>APG</th><th>SPG</th><th>FG%</th></tr>
    <tr><td>Regular Season</td><td>1,072</td><td>30.1</td><td>6.2</td><td>5.3</td><td>2.3</td><td>49.7%</td></tr>
    <tr><td>Playoffs</td><td>179</td><td>33.4</td><td>6.4</td><td>5.7</td><td>2.1</td><td>48.7%</td></tr>
    <tr><td>All-Star Games</td><td>14</td><td>20.5</td><td>5.9</td><td>3.8</td><td>2.2</td><td>51.0%</td></tr>
  </tbody>
</table>

<ac:structured-macro ac:name="info">
  <ac:parameter ac:name="title">All-Time Records</ac:parameter>
  <ac:rich-text-body>
    <ul>
      <li>NBA all-time highest career scoring average: <strong>30.12 PPG</strong></li>
      <li>10 scoring titles — most in NBA history</li>
      <li>3 steals titles</li>
      <li>Consecutive games scoring 10+ points: 866 (NBA record)</li>
    </ul>
  </ac:rich-text-body>
</ac:structured-macro>

<h2>2. Early Life and College Career</h2>
<p>Jordan grew up in Wilmington, North Carolina. He was famously cut from his varsity high school team as a sophomore, a rejection he later credited as a pivotal motivator. He attended the <strong>University of North Carolina at Chapel Hill</strong> (1981–1984), where he played under coach Dean Smith.</p>

<table>
  <tbody>
    <tr><th>Year</th><th>Event</th></tr>
    <tr><td>1982</td><td>Hit the game-winning jump shot in the NCAA Championship against Georgetown as a freshman</td></tr>
    <tr><td>1983</td><td>Named ACC Player of the Year</td></tr>
    <tr><td>1984</td><td>Won the Naismith and Wooden Awards (National Player of the Year); left UNC after junior year</td></tr>
    <tr><td>1984</td><td>Selected 3rd overall in the NBA Draft by the Chicago Bulls</td></tr>
  </tbody>
</table>

<h2>3. NBA Career Timeline</h2>

<h3>3.1 Chicago Bulls — First Run (1984–1993)</h3>
<p>Jordan's first stint with the Bulls transformed a losing franchise into a dynasty. After early playoff exits, the arrival of coach Phil Jackson and the triangle offense unlocked the team's potential.</p>

<table>
  <tbody>
    <tr><th>Season</th><th>Achievement</th></tr>
    <tr><td>1984–85</td><td>Rookie of the Year; 28.2 PPG</td></tr>
    <tr><td>1985–86</td><td>Returned from broken foot; scored 63 points vs. Boston Celtics in playoffs (single-game playoff record)</td></tr>
    <tr><td>1987–88</td><td>First MVP award; Defensive Player of the Year; first of 7 consecutive scoring titles</td></tr>
    <tr><td>1990–91</td><td>First NBA Championship; first Finals MVP; averaged 31.2 PPG in Finals vs. LA Lakers</td></tr>
    <tr><td>1991–92</td><td>Second championship; famous "flu game" in 1993 Finals</td></tr>
    <tr><td>1992–93</td><td>Third championship; first three-peat; scored 41 PPG in Finals vs. Phoenix Suns</td></tr>
  </tbody>
</table>

<h3>3.2 First Retirement and Baseball (1993–1995)</h3>
<p>In October 1993, following the murder of his father James Jordan, Michael Jordan shocked the world by retiring from basketball to pursue a professional baseball career, honoring his father's dream for him.</p>
<ul>
  <li>Signed a minor league contract with the <strong>Chicago White Sox</strong> organization</li>
  <li>Played for the <strong>Birmingham Barons</strong> (Double-A) in 1994: .202 BA, 3 HR, 51 RBI in 127 games</li>
  <li>Invited to 1995 spring training with White Sox before returning to the NBA</li>
</ul>

<h3>3.3 Chicago Bulls — Second Run (1995–1998)</h3>
<p>Jordan returned to the Bulls on March 18, 1995, with a two-word fax: "I'm back." The second three-peat is considered the peak of his dominance.</p>

<table>
  <tbody>
    <tr><th>Season</th><th>Achievement</th></tr>
    <tr><td>1995–96</td><td>72-10 regular season record (then-NBA record); 4th MVP; 4th championship</td></tr>
    <tr><td>1996–97</td><td>5th championship; "flu game" vs. Utah Jazz (Game 5, 38 points despite severe illness)</td></tr>
    <tr><td>1997–98</td><td>6th championship; "The Last Shot" vs. Utah Jazz to clinch title; 5th MVP; final season with Bulls</td></tr>
  </tbody>
</table>

<ac:structured-macro ac:name="note">
  <ac:parameter ac:name="title">The 1995–96 Bulls</ac:parameter>
  <ac:rich-text-body>
    <p>The 1995–96 Chicago Bulls finished 72-10, a record that stood for 20 years until the Golden State Warriors went 73-9 in 2015–16. The Bulls roster included Jordan, Scottie Pippen, Dennis Rodman, Toni Kukoc, and Steve Kerr, coached by Phil Jackson.</p>
  </ac:rich-text-body>
</ac:structured-macro>

<h3>3.4 Second Retirement and Washington Wizards (1999–2003)</h3>
<p>Jordan retired again in January 1999, became part-owner and President of Basketball Operations for the Washington Wizards, then made a final comeback as a player at age 38.</p>

<table>
  <tbody>
    <tr><th>Season</th><th>Stats</th><th>Notes</th></tr>
    <tr><td>2001–02</td><td>22.9 PPG, 5.7 RPG, 5.2 APG</td><td>Age 38; led Wizards in scoring</td></tr>
    <tr><td>2002–03</td><td>20.0 PPG, 6.1 RPG, 3.8 APG</td><td>Final NBA season; scored 43 points in final game in Philadelphia</td></tr>
  </tbody>
</table>

<h2>4. International Career</h2>
<table>
  <tbody>
    <tr><th>Year</th><th>Team</th><th>Result</th></tr>
    <tr><td>1984</td><td>USA Olympic Team (Los Angeles)</td><td>Gold Medal</td></tr>
    <tr><td>1992</td><td>USA "Dream Team" (Barcelona)</td><td>Gold Medal; averaged 14.9 PPG</td></tr>
  </tbody>
</table>

<h2>5. Awards and Honours</h2>
<table>
  <tbody>
    <tr><th>Award</th><th>Times Won</th><th>Years</th></tr>
    <tr><td>NBA Champion</td><td>6</td><td>1991, 1992, 1993, 1996, 1997, 1998</td></tr>
    <tr><td>Finals MVP</td><td>6</td><td>1991, 1992, 1993, 1996, 1997, 1998</td></tr>
    <tr><td>Regular Season MVP</td><td>5</td><td>1988, 1991, 1992, 1996, 1998</td></tr>
    <tr><td>Scoring Title</td><td>10</td><td>1987–1993, 1996, 1997, 1998</td></tr>
    <tr><td>Defensive Player of the Year</td><td>1</td><td>1988</td></tr>
    <tr><td>Rookie of the Year</td><td>1</td><td>1985</td></tr>
    <tr><td>All-NBA First Team</td><td>10</td><td>1987–1993, 1996, 1997, 1998</td></tr>
    <tr><td>NBA All-Defensive First Team</td><td>9</td><td>1988–1993, 1996, 1997, 1998</td></tr>
    <tr><td>Olympic Gold Medal</td><td>2</td><td>1984, 1992</td></tr>
    <tr><td>NBA 50th Anniversary Team</td><td>—</td><td>1996</td></tr>
    <tr><td>NBA 75th Anniversary Team</td><td>—</td><td>2021</td></tr>
    <tr><td>Naismith Hall of Fame</td><td>—</td><td>2009</td></tr>
  </tbody>
</table>

<h2>6. Legacy and Cultural Impact</h2>
<p>Beyond statistics, Jordan's impact on basketball and popular culture is unmatched:</p>
<ul>
  <li><strong>Air Jordan / Nike:</strong> Launched in 1985, the Air Jordan sneaker line generates over $5 billion annually and redefined athlete endorsements.</li>
  <li><strong>Space Jam (1996):</strong> The film grossed $230 million worldwide and cemented Jordan's crossover pop-culture status.</li>
  <li><strong>The Last Dance (2020):</strong> Netflix/ESPN documentary about the 1997–98 Bulls became the most-watched ESPN documentary ever.</li>
  <li><strong>Charlotte Hornets ownership:</strong> Jordan became the first former NBA player to become the majority owner of an NBA team (2010), selling his majority stake in 2023.</li>
</ul>

<ac:structured-macro ac:name="info">
  <ac:parameter ac:name="title">GOAT Debate</ac:parameter>
  <ac:rich-text-body>
    <p>The debate over the greatest basketball player of all time most commonly places Jordan alongside LeBron James. Jordan's six championships with zero Finals losses, ten scoring titles, and Defensive Player of the Year award form the core of his case. His career regular-season scoring average of 30.12 PPG remains the highest in NBA history.</p>
  </ac:rich-text-body>
</ac:structured-macro>
"""

payload = {
    "type": "page",
    "title": "Michael Jordan — Career Overview",
    "space": {"key": SPACE_KEY},
    "ancestors": [{"id": PARENT_ID}],
    "body": {
        "storage": {
            "representation": "storage",
            "value": PAGE_BODY,
        }
    },
}

resp = requests.post(
    f"{BASE}/rest/api/content",
    headers=HEADERS,
    data=json.dumps(payload),
)

if resp.status_code in (200, 201):
    d = resp.json()
    print(f"Created page ID : {d['id']}")
    print(f"Title           : {d['title']}")
    print(f"URL             : {BASE}{d['_links']['webui']}")
else:
    print(f"Error {resp.status_code}:")
    print(resp.text[:1000])
