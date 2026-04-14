import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Football Knowledge Base",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Inter:wght@300;400;600&display=swap');

html, body, [class*="css"] {
    background-color: #0d1117;
    color: #e6edf3;
    font-family: 'Inter', sans-serif;
}
.stApp { background-color: #0d1117; }

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}
section[data-testid="stSidebar"] * { color: #e6edf3 !important; }

h1 { font-family: 'Bebas Neue', sans-serif !important; font-size: 3.2rem !important;
     letter-spacing: 3px; color: #58a6ff !important; margin-bottom: 0 !important; }
h2 { font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important;
     letter-spacing: 2px; color: #79c0ff !important; }
h3 { font-family: 'Bebas Neue', sans-serif !important; font-size: 1.4rem !important;
     letter-spacing: 1px; color: #d2a679 !important; }

.stat-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 18px 20px;
    text-align: center;
}
.stat-card .stat-value {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 2.2rem;
    color: #58a6ff;
    line-height: 1.1;
}
.stat-card .stat-label {
    color: #8b949e;
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
}

.stTextInput > div > div > input {
    background: #161b22 !important;
    border: 1.5px solid #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
    font-size: 1rem !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}

.stButton > button {
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: #ffffff;
    border: none;
    border-radius: 8px;
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 1.5px;
    font-size: 1.1rem;
    padding: 10px 28px;
    transition: all 0.2s ease;
}
.stButton > button:hover {
    background: linear-gradient(135deg, #388bfd, #58a6ff);
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(88,166,255,0.35);
}

.result-card {
    background: #161b22;
    border: 1px solid #21262d;
    border-left: 4px solid #58a6ff;
    border-radius: 10px;
    padding: 20px 24px;
    margin-bottom: 16px;
    transition: border-color 0.2s;
}
.result-card:hover { border-left-color: #d2a679; }
.result-rank {
    font-family: 'Bebas Neue', sans-serif;
    color: #58a6ff;
    font-size: 0.9rem;
    letter-spacing: 2px;
    margin-bottom: 6px;
}
.result-text { color: #c9d1d9; line-height: 1.7; font-size: 0.95rem; }

.score-badge {
    display: inline-block;
    background: rgba(88,166,255,0.15);
    border: 1px solid rgba(88,166,255,0.4);
    color: #58a6ff;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin-top: 10px;
}

.info-box {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 20px 24px;
    margin: 16px 0;
}

.topic-pill {
    display: inline-block;
    background: rgba(210,166,121,0.15);
    border: 1px solid rgba(210,166,121,0.4);
    color: #d2a679;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.8rem;
    margin: 3px;
}

hr { border-color: #21262d !important; }
.stSelectbox > div > div, .stSlider { color: #e6edf3 !important; }
.streamlit-expanderHeader { color: #79c0ff !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
#  DOCUMENTS
# ─────────────────────────────────────────────
DOCUMENTS = [
    """Football (or soccer as the game is called in some parts of the world) has a long history. Football in its current form arose in England in the middle of the 19th century. But alternative versions of the game existed much earlier and are a part of the football history.

The first known examples of a team game involving a ball, which was made out of a rock, occurred in old Mesoamerican cultures for over 3,000 years ago. It was by the Aztecs called Tchatali, although various versions of the game were spread over large regions. In some ritual occasions, the ball would symbolize the sun and the captain of the losing team would be sacrificed to the gods. A unique feature of the Mesoamerican ball game versions was a bouncing ball made of rubber – no other early culture had access to rubber.

The first known ball game which also involved kicking took place In China in the 3rd and 2nd century BC under the name cuju. Cuju was played with a round ball (stitched leather with fur or feathers inside) on an area of a square. A modified form of this game later spread to Japan and was by the name of kemari practiced under ceremonial forms.""",

    """The aim of football is to score more goals then your opponent in a 90 minute playing time frame. The match is split up into two halves of 45 minutes. After the first 45 minutes players will take a 15 minute rest period called half time. The second 45 minutes will resume and any time deemed fit to be added on by the referee (injury time) will be accordingly.

Each team consists of 11 players. These are made up of one goalkeeper and ten outfield players. The pitch dimensions vary from each ground but are roughly 120 yards long and 75 yards wide. On each pitch you will have a 6 yard box next to the goal mouth, an 18 yard box surrounding the 6 yard box and a centre circle. Each half of the pitch must be a mirror image of the other in terms of dimensions.

Essentially the equipment that is needed for a soccer match is pitch and a football. Additionally players can be found wearing studded football boots, shin pads and matching strips. The goalkeepers will additionally wear padded gloves as they are the only players allowed to handle the ball. Each team will have a designated captain.""",

    """The FIFA World Cup, often called the World Cup, is an international association football competition among the senior men's national teams of the members of the Fédération Internationale de Football Association (FIFA), the sport's global governing body. The tournament has been held every four years since the inaugural tournament in 1930, with the exception of 1942 and 1946 due to the Second World War. The reigning champions are Argentina, who won their third title at the 2022 World Cup by defeating France.

The contest starts with the qualification phase, which takes place over the preceding three years to determine which teams qualify for the tournament phase. In the tournament phase, 32 teams compete for the title at venues within the host nation(s) over the course of about a month. The host nation(s) automatically qualify for the group stage of the tournament. The competition is scheduled to expand to 48 teams, starting with the 2026 World Cup.

As of the 2022 World Cup, 22 final tournaments have been held since the event's inception in 1930, and a total of 80 national teams have competed. The trophy has been won by eight national teams. With five wins, Brazil is the only team to have played in every tournament. The other World Cup winners are Germany and Italy, with four titles each; Argentina, with three titles; France and inaugural winner Uruguay, each with two titles; and England and Spain, with one title each.""",

    """The Premier League is a professional association football league in England and the highest level of the English football league system. Contested by 20 clubs, it operates on a system of promotion and relegation with the English Football League (EFL). Seasons usually run from August to May, with each team playing 38 matches: two against each other team, one home and one away. Most games are played on weekend afternoons, with occasional weekday evening fixtures.

The competition was founded as the FA Premier League on 20 February 1992, following the decision of clubs from the First Division (the top tier since 1888) to break away from the English Football League. Teams are still promoted and relegated to and from the EFL Championship each season. The Premier League is a corporation managed by a chief executive, with member clubs as shareholders. The Premier League takes advantage of a £5 billion domestic television rights deal, with Sky and BT Group broadcasting 128 and 32 games, respectively. This will rise to £6.7 billion from 2025 to 2029.

The Premier League is the most-watched sports league in the world, broadcast in 212 territories to 643 million homes, with a potential TV audience of 4.7 billion people. As of the 2024–25 season, the Premier League has the highest average and aggregate match attendance of any association football league in the world, at 40,421 per game.""",

    """Lionel Andrés "Leo" Messi (born 24 June 1987) is an Argentine professional footballer who plays as a forward for and captains both Major League Soccer club Inter Miami and the Argentina national team. Widely regarded as one of the greatest players in history, Messi has set numerous records for individual accolades won throughout his professional footballing career, including eight Ballon d'Ors, six European Golden Shoes, and eight times being named the world's best player by FIFA.

Messi made his competitive debut for Barcelona at age 17 in October 2004. He gradually established himself as an integral player for the club, and during his first uninterrupted season in 2008–09 he helped Barcelona achieve the first treble in Spanish football. This resulted in Messi winning the first of four consecutive Ballon d'Ors, and by the 2011–12 season he had set the European record for most goals in a season. During the 2014–15 season, where he became the all-time top scorer in La Liga, he led Barcelona to a historic second treble.

While playing for Barcelona, he won a club-record 34 trophies, including ten La Liga titles and four UEFA Champions Leagues. Financial difficulties at Barcelona led Messi to depart in August 2021 and sign with Paris Saint-Germain. He joined the Major League Soccer club Inter Miami in July 2023, and led them to their first MLS Cup victory in 2025.""",

    """Cristiano Ronaldo dos Santos Aveiro (born 5 February 1985), nicknamed CR7, is a Portuguese professional footballer who plays as a forward for and captains both Saudi Pro League club Al-Nassr and the Portugal national team. Widely regarded as one of the greatest players in history, he has won numerous individual accolades throughout his career, including five Ballon d'Ors, a record three UEFA Men's Player of the Year Awards, and four European Golden Shoes.

Born in Funchal, Madeira, Ronaldo began his career with Sporting CP before signing with Manchester United in 2003. He became a star player at United, where he won three consecutive Premier League titles, the Champions League, and the FIFA Club World Cup. In 2009, Ronaldo became the subject of the then-most expensive transfer in history when he joined Real Madrid in a deal worth €94 million. At Madrid, he helped win four Champions Leagues between 2014 and 2018, including the long-awaited La Décima.

He holds the records for most goals (140) and assists (42) in the Champions League, goals (14) and assists (8) in the European Championship, and most international appearances (226) and international goals (143). He is the only player to have scored 100 goals with four different clubs and has scored over 960 official senior career goals for club and country, making him the top goalscorer of all time.""",

    """The 4-3-3 is a formation that uses four defenders – made up of two centre-backs and two full-backs – behind a midfield line of three. The most common set-up in midfield is one deeper player – the single pivot – and two slightly more advanced to either side. The front line is then composed of two wide attackers who play on either side of a single centre-forward.

In response to the 1950 World Cup final defeat at home to Uruguay, Brazil started to use a back line of four in a 4-2-4 formation. By 1962, Brazil had adapted once more to create a 4-3-3 structure in which Mário Zagallo dropped from the front line into midfield.

Rinus Michels' Netherlands and Ajax sides of the 1970s were famous for inspiring the use of a 4-3-3 shape. It was these teams' tactics that led to the concept of Total Football and encouraged the likes of Johan Cruyff to use a 4-3-3 when he became a coach. The main responsibility for the wingers is isolating full-backs and attacking in one-on-one situations – either working around the outside of their opponents to cross, or cutting inside to combine.""",

    """Women's association football, more commonly known as women's football or women's soccer, is the team sport of association football played by women. It is played at the professional level in multiple countries, and about 200 national teams participate internationally. The United States is the most successful nation in women's football, having won 4 Women's World Cups and 5 Olympic gold medals. The second most successful nation in women's football is Germany, with 2 Women's World Cups and 1 Olympic gold medal.

During the 1970s, Italy became the first country to have professional women's football players on a part-time basis. Italy was also the first country to import foreign footballers from other European countries, which raised the profile of the league. Sweden was the first to introduce a professional women's domestic league in 1988, the Damallsvenskan.

A 2014 FIFA report stated that at the beginning of the 21st century, women's football was growing in both popularity and participation, and more professional leagues were being launched worldwide. From the inaugural FIFA Women's World Cup tournament held in 1991 to the 1,194,221 tickets sold for the 1999 Women's World Cup, visibility and support of women's professional football had increased around the globe.""",

    """Football is one of the most popular sports in the world, both for cheering and playing. Players tend to collide into each other on the field and injuries can occur at any time, especially towards the end of play when fatigue sets in. The most common football injury is muscle injury, but other serious injuries include torn ligament, broken ankle, or concussion, which will require long term care.

If the players are not physically ready in their fitness and connective tissues, or imbalanced homeostasis, it may lead to injuries during practice or competition. Warming up and cooling down before and after playing sports will help condition the muscles and tendons to have more flexibility. FIFA 11+ warm up program which has been well designed can reduce injuries by up to 30%.

When choosing football shoes, suitability should be kept in mind. If improper shoes are used, it may cause a slip and injury to the player. Currently, football fields with imitation grass are popular in many countries. These synthetic grasses have more friction than real grass. If the player chooses studs or blades, there might be too much friction leading to serious injuries. Same thing with protective gears, such as shin guards, which can reduce force during contact and reduce serious injuries.""",

    """The UEFA Champions League (UCL), commonly known as the Champions League, is an annual club association football competition organised by UEFA that is contested by top-division European clubs. It is the most-watched club competition in the world and the third most-watched football competition overall, behind only the FIFA World Cup and the UEFA European Championship.

In its present format, the Champions League begins in early July with qualifying rounds. The 36 teams each play eight opponents, four home and four away. The 24 highest-ranked teams proceed to the knockout phase that culminates with the final match in late May or early June. The winner automatically qualifies for the following year's Champions League, the UEFA Super Cup, the FIFA Intercontinental Cup and the FIFA Club World Cup.

Real Madrid is the most successful club in the tournament's history, having won it 15 times. Madrid is the only club to have won it five times in a row (the first five editions). Only one club has won all of their matches in a single tournament en route to their victory: Bayern Munich in the 2019–20 season. Paris Saint-Germain are the current European champions, having beaten Inter Milan 5–0 in the 2025 final for their first ever title.""",

    """The Brazil national football team, nicknamed Seleção Canarinho ("Canary Squad", after their bright yellow jersey), represents Brazil in men's international football and is administered by the Confederação Brasileira de Futebol. It has been a member of FIFA since 1923 and a founding member of CONMEBOL since 1916.

Brazil is the most successful national team in the FIFA World Cup, winning the tournament five times: 1958, 1962, 1970, 1994 and 2002. It is the only national team to have played in all World Cup editions without any absence nor need for playoffs, and the only team to have won the World Cup in four different continents. Brazil was also the most successful team in the now-defunct FIFA Confederations Cup, winning it four times.

Brazil has the highest average Elo football rating over time. Many commentators, experts, and former players have considered the Brazil team of 1970, led by Pelé, to be the greatest team of all time. In 1996, the Brazil national team achieved 35 consecutive matches undefeated, a feat which they held as a world record for 25 years.""",

    """In association football, a transfer window is the period during the year in which a club can add players to their squad who were previously under contract with another club. Such a transfer is completed by registering the player into the new club through FIFA. According to the rules, each national football association decides on the time of the 'window' but it may not exceed 12 weeks. The second registration period occurs during the season and may not exceed four weeks.

The transfer window of a given football association governs only international transfers into that football association. International transfers out of an association are always possible to those associations that have an open window. The window was introduced in response to negotiations with the European Commission.

The system had been used in many European leagues before being brought into compulsory effect by FIFA during the 2002–03 season. English football was initially behind the plans when they were proposed in the early 1990s, in the hope that it would improve teams' stability and prevent agents from searching for deals all year around. The exact regulations and possible exceptions are established by each competition's governing body rather than by the national football association.""",
]

TOPICS = [
    "History of Football", "Rules & Gameplay", "FIFA World Cup",
    "Premier League", "Lionel Messi", "Cristiano Ronaldo",
    "Tactics & Formations", "Women's Football", "Injuries & Safety",
    "Champions League", "Brazil National Team", "Transfer Windows",
]

# ─────────────────────────────────────────────
#  SEARCH INDEX
# ─────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_index(chunk_size: int, chunk_overlap: int):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    all_chunks = []
    all_metadatas = []
    for i, text in enumerate(DOCUMENTS):
        chunks = splitter.split_text(text)
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({"topic": TOPICS[i], "doc_index": i})

    vectorizer = TfidfVectorizer()
    matrix = vectorizer.fit_transform(all_chunks)
    return vectorizer, matrix, all_chunks, all_metadatas, len(all_chunks)


def search(vectorizer, matrix, all_chunks, all_metadatas, query: str, n_results: int):
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, matrix)[0]
    top_indices = np.argsort(scores)[::-1][:n_results]
    return [(all_chunks[i], all_metadatas[i], float(scores[i])) for i in top_indices]


# ─────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚽ FOOTBALL KB")
    st.markdown("---")
    page = st.radio(
        "Navigate",
        ["🏠 Home", "🔍 Search", "📊 Stats & Info"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown("### ⚙️ Chunking Settings")
    chunk_size = st.select_slider(
        "Chunk Size",
        options=[100, 200, 300, 500, 700, 1000],
        value=300,
        help="Characters per chunk. Smaller = more precise; larger = more context.",
    )
    chunk_overlap = st.select_slider(
        "Chunk Overlap",
        options=[0, 20, 50, 80, 100],
        value=50,
        help="Overlap between consecutive chunks to preserve context at boundaries.",
    )
    n_results = st.slider("Results to show", 1, 8, 4)
    st.markdown("---")
    st.markdown(
        "<div style='color:#8b949e;font-size:0.78rem;'>Model: TF-IDF + Cosine Similarity<br>"
        "Library: scikit-learn</div>",
        unsafe_allow_html=True,
    )

# ─────────────────────────────────────────────
#  BUILD INDEX
# ─────────────────────────────────────────────
with st.spinner("🔧 Building search index…"):
    vectorizer, matrix, all_chunks, all_metadatas, total_chunks = build_index(chunk_size, chunk_overlap)

# ═════════════════════════════════════════════
#  PAGE: HOME
# ═════════════════════════════════════════════
if page == "🏠 Home":
    st.markdown("<h1>⚽ FOOTBALL<br>KNOWLEDGE BASE</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;font-size:1.05rem;margin-top:-8px;'>"
        "Semantic search across the beautiful game — history, tactics, players, competitions & more."
        "</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Stat cards (pure HTML, no st.metric) ──
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:24px;">
      <div class="stat-card"><div class="stat-value">{len(DOCUMENTS)}</div><div class="stat-label">📄 Documents</div></div>
      <div class="stat-card"><div class="stat-value">{total_chunks}</div><div class="stat-label">🧩 Chunks</div></div>
      <div class="stat-card"><div class="stat-value">{chunk_size}</div><div class="stat-label">📐 Chunk Size (chars)</div></div>
      <div class="stat-card"><div class="stat-value">{chunk_overlap}</div><div class="stat-label">🔗 Overlap (chars)</div></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📚 Topics Covered")
    pills_html = "".join(f'<span class="topic-pill">{t}</span>' for t in TOPICS)
    st.markdown(f'<div class="info-box">{pills_html}</div>', unsafe_allow_html=True)

    st.markdown("### 🚀 How It Works")
    st.markdown("""
<div class="info-box">
<ol style="color:#c9d1d9;line-height:2;margin:0;padding-left:20px;">
  <li><b style="color:#58a6ff;">Chunking</b> — Each document is split into overlapping text chunks using <code>RecursiveCharacterTextSplitter</code>.</li>
  <li><b style="color:#58a6ff;">Vectorising</b> — Chunks are converted into TF-IDF vectors which capture word importance across the corpus.</li>
  <li><b style="color:#58a6ff;">Indexing</b> — Vectors are stored in memory using scikit-learn.</li>
  <li><b style="color:#58a6ff;">Retrieval</b> — Your query is vectorised and the closest chunks are returned by cosine similarity.</li>
</ol>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 💡 Try These Queries")
    example_queries = [
        "Who has scored the most Champions League goals?",
        "How does the offside rule work?",
        "What is the history of the World Cup?",
        "Tell me about Messi's career at Barcelona",
        "What injuries are common in football?",
        "How does the 4-3-3 formation work?",
    ]
    cols = st.columns(2)
    for i, q in enumerate(example_queries):
        cols[i % 2].markdown(
            f'<div class="info-box" style="padding:12px 16px;margin:6px 0;">'
            f'<span style="color:#58a6ff;">🔎</span> '
            f'<span style="color:#c9d1d9;font-size:0.9rem;">{q}</span></div>',
            unsafe_allow_html=True,
        )

# ═════════════════════════════════════════════
#  PAGE: SEARCH
# ═════════════════════════════════════════════
elif page == "🔍 Search":
    st.markdown("<h1>🔍 SEMANTIC SEARCH</h1>", unsafe_allow_html=True)
    st.markdown(
        "<p style='color:#8b949e;'>Ask anything about football — the engine finds the most relevant passages.</p>",
        unsafe_allow_html=True,
    )
    st.markdown("---")

    query = st.text_input(
        "Your question",
        placeholder="e.g. Who won the 2022 World Cup? / What is the offside rule?",
        label_visibility="collapsed",
    )

    col_btn, _ = st.columns([1, 5])
    search_clicked = col_btn.button("⚽ Search")

    if search_clicked and query.strip():
        with st.spinner("Searching the knowledge base…"):
            results = search(vectorizer, matrix, all_chunks, all_metadatas, query.strip(), n_results)

        if results:
            st.markdown(f"### Found {len(results)} relevant passage(s) for: *\"{query}\"*")
            st.markdown("")
            for rank, (text, meta, score) in enumerate(results, 1):
                topic = meta.get("topic", "Unknown")
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-rank">#{rank} — {topic}</div>'
                    f'<div class="result-text">{text}</div>'
                    f'<span class="score-badge">Relevance: {score:.2%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning("No results found. Try a different query.")

    elif search_clicked:
        st.warning("Please enter a query first.")

    st.markdown("---")
    st.markdown("#### ⚡ Quick searches")
    quick = [
        "Messi Ballon d'Or", "World Cup winners", "Premier League history",
        "Football injuries", "Transfer window rules", "Women's World Cup",
    ]
    cols = st.columns(3)
    for i, q in enumerate(quick):
        if cols[i % 3].button(q, key=f"quick_{i}"):
            with st.spinner("Searching…"):
                results = search(vectorizer, matrix, all_chunks, all_metadatas, q, n_results)
            st.markdown(f"### Results for: *\"{q}\"*")
            for rank, (text, meta, score) in enumerate(results, 1):
                topic = meta.get("topic", "Unknown")
                st.markdown(
                    f'<div class="result-card">'
                    f'<div class="result-rank">#{rank} — {topic}</div>'
                    f'<div class="result-text">{text}</div>'
                    f'<span class="score-badge">Relevance: {score:.2%}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

# ═════════════════════════════════════════════
#  PAGE: STATS & INFO
# ═════════════════════════════════════════════
elif page == "📊 Stats & Info":
    st.markdown("<h1>📊 STATS & INFO</h1>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 🔬 Chunking Strategy Comparison")
    st.markdown("""
<div class="info-box">
<table style="width:100%;border-collapse:collapse;color:#c9d1d9;font-size:0.9rem;">
  <thead>
    <tr style="border-bottom:1px solid #30363d;">
      <th style="text-align:left;padding:10px;color:#58a6ff;">Chunk Size</th>
      <th style="text-align:left;padding:10px;color:#58a6ff;">Overlap</th>
      <th style="text-align:left;padding:10px;color:#58a6ff;">Best For</th>
      <th style="text-align:left;padding:10px;color:#58a6ff;">Trade-off</th>
    </tr>
  </thead>
  <tbody>
    <tr style="border-bottom:1px solid #21262d;">
      <td style="padding:10px;">100–200 chars</td>
      <td style="padding:10px;">20</td>
      <td style="padding:10px;">Precise fact retrieval</td>
      <td style="padding:10px;">May lose surrounding context</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d;">
      <td style="padding:10px;"><b style="color:#d2a679;">300 chars ✓</b></td>
      <td style="padding:10px;">50</td>
      <td style="padding:10px;">Balanced precision & context</td>
      <td style="padding:10px;">Good default for this dataset</td>
    </tr>
    <tr style="border-bottom:1px solid #21262d;">
      <td style="padding:10px;">500–700 chars</td>
      <td style="padding:10px;">80–100</td>
      <td style="padding:10px;">Narrative / contextual answers</td>
      <td style="padding:10px;">Less precise on specific facts</td>
    </tr>
    <tr>
      <td style="padding:10px;">1000 chars</td>
      <td style="padding:10px;">100</td>
      <td style="padding:10px;">Long-form summaries</td>
      <td style="padding:10px;">Dilutes relevance signal</td>
    </tr>
  </tbody>
</table>
</div>
""", unsafe_allow_html=True)

    st.markdown("### 📋 Document Inventory")
    for i, (doc, topic) in enumerate(zip(DOCUMENTS, TOPICS), 1):
        word_count = len(doc.split())
        with st.expander(f"#{i:02d}  {topic}  —  {word_count} words"):
            st.write(doc[:600] + ("…" if len(doc) > 600 else ""))

    st.markdown("### 🧠 Search Method")
    st.markdown("""
<div class="info-box">
<ul style="color:#c9d1d9;line-height:2;margin:0;padding-left:20px;">
  <li><b style="color:#58a6ff;">Method:</b> TF-IDF (Term Frequency–Inverse Document Frequency)</li>
  <li><b style="color:#58a6ff;">Similarity:</b> Cosine similarity between query and chunk vectors</li>
  <li><b style="color:#58a6ff;">Library:</b> scikit-learn (no model download required)</li>
  <li><b style="color:#58a6ff;">Memory:</b> Extremely lightweight — runs on Render free tier easily</li>
  <li><b style="color:#58a6ff;">Trade-off:</b> Keyword-based rather than semantic — great for factual queries</li>
</ul>
</div>
""", unsafe_allow_html=True)