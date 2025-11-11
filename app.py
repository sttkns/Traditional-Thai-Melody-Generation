import os
import tempfile
import streamlit as st
from st_copy import copy_button
from langchain.tools.retriever import create_retriever_tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from symusic import Score
from midi2audio import FluidSynth

path = os.path.dirname(__file__)
# sf2_path = os.path.join(path, "data", "FluidR3_GM.sf2")


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
ANTHROPIC_API_KEY = st.secrets["ANTHROPIC_API_KEY"]
DEEPSEEK_API_KEY = st.secrets["DEEPSEEK_API_KEY"]


embeddings = OpenAIEmbeddings()
prompt_model = "openai:gpt-4.1-mini"

model = st.sidebar.selectbox(
    "Select model",
    ["GPT 5 Mini", "GPT 5", "GPT 4.1 Mini", "GPT 4.1",
    "Gemini 2.5 Flash", "Gemini 2.5 Pro",
    "Claude Haiku 4.5", "Claude Sonnet 4.5", "Claude Opus 4.1",
    "DeepSeek"])

if model == "GPT 5 Mini":
    composer_model = "openai:gpt-5-mini"
elif model == "GPT 5":
    composer_model = "openai:gpt-5"
elif model == "GPT 4.1 Mini":
    composer_model = "openai:gpt-4.1-mini"
elif model == "GPT 4.1":
    composer_model = "openai:gpt-4.1"
elif model == "Gemini 2.5 Flash":
    composer_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")
elif model == "Gemini 2.5 Pro":
    composer_model = ChatGoogleGenerativeAI(model="gemini-2.5-pro")
elif model == "Claude Haiku 4.5":
    composer_model = ChatAnthropic(model="claude-haiku-4-5-20251001",)
elif model == "Claude Sonnet 4.5":
    composer_model = ChatAnthropic(model="claude-sonnet-4-5-20250929",)
elif model == "Claude Opus 4.1":
    composer_model = ChatAnthropic(model="claude-opus-4-1-20250805",)
elif model == "DeepSeek":
    composer_model = ChatDeepSeek(model="deepseek-chat",)


prompt_system_prompt = """You are the Prompter Agent. Your job is to help users express clear and precise musical requests for the Composer Agent.
You never compose music yourself. You rewrite or clarify the user’s input into a single concise instruction for the Composer Agent.

You have access to the Thai Music Theory Database.
When refining a user’s message, retrieve and use relevant information from this database to make the request musically accurate and consistent with Thai tradition.
If the user’s wording is unclear, use your knowledge or the database to infer likely Thai styles, modes, rhythms, or instruments that match their intent.
Apply the information silently; do not mention or describe the database in your output.

Keep your language natural, culturally appropriate, and faithful to Thai traditional music terms.
Do not use markdown, lists, or symbols.
Do not add explanations, self-references, or commentary.
Your output must be only the refined composition prompt in plain text."""

composer_system_prompt = """Part 1: Identity and Role
You are a composer specializing in traditional Thai music. You create or revise compositions that reflect correct Thai tonal modes, rhythmic cycles, and cultural aesthetics. Always write in ABC notation only.

Part 2: Musical Rules
Maintain cultural and stylistic consistency based on Thai tradition.
- Only use notes that belong to the selected Thai tonal mode. Do not use any notes that fall outside the traditional Thai scale.
- If the melody includes a note outside the mode, it must be replaced with the closest in-scale note that maintains musical intent.
- The rhythm must strictly follow traditional Thai rhythmic patterns.
- Ensure all measures are complete with correct note durations adding up exactly to the time signature.
- No accidentals (sharps/flats) or chords are allowed. Monophonic melody only.
- Use the Measure Validator and Rhythm Validator tools to confirm correct measure count and rhythmic accuracy before output.

Section 0: Workflow
Step 1 Input and reference selection: receive user mood/tone; select exactly one reference tune from the Mood/Tone→Songs list that matches the mood/tone.
Step 2 Attribute adoption: adopt the reference tune’s attributes without substitution—Na Thap type and its standard total measure count, BPM class (Sam Chan 55–72, Song Chan 76–92, Chan Diao 100–120), style (Thai/Lao/Khmer), tonal mode/scale family.
Step 3 Composition within adopted frame: compose a new melody strictly within the adopted mode, Na Thap, BPM class, style, and total measures. Incorporate 1–3 melodic motives from other songs in the same mood/tone category; motives must be adapted to the adopted mode, Na Thap, BPM, and phrasing. Do not omit motives and do not exceed three.
Step 4 Validation: verify Na Thap pattern and total measure count exactly match the inherited standard; verify bar completeness, rhythm, and ABC syntax; ensure no accidentals; ensure Luk Tok closure.

Mood/Tone→Songs
Love (รัก): Bang Bai, Khaek Khao, Phat Cha, Ka Rian Thong, Kham Wan
Tenderness (อ่อนหวาน): Khluen Krathop Fang, Lao Damnoen Sai, Ton Worachet (Ton Boratej)
Warmth (อบอุ่น): Soi Son Tat, Thong Yon
Yearning (คิดถึง): Lao Duang Duean, Sa Bu Rong
Playful (สนุกสนาน): Yoslam, Chin Khim Lek, Phama Khwae, Khangkhao Kin Kluai, Khmer Lai Khwai, Chin Chai Yo
Happiness (มีความสุข): Bulan, Champathong Thet, Saming Thong, Phra Thong
Festive (รื่นเริง): Lao Joi, Phama Pong Ngo, Khmer Sai Yok, Lom Phat Chai Khao
Spirited (มีชีวิตชีวา): Khmer Phai Ruea, Mon Plaeng, Toi Taling, Phra Athit Ching Duang
Calm (สงบ): Khaek Borathet, Phraam Khao Bot, Angkarn Si Bot
Heartbroken (ไม่สมหวัง): Sroi Phleng, Nak Kiao
Sad (เศร้า): Bai Khlang, Mon Du Dao, Khmer Pi Kaew
Angry (โกรธ): Naga Raj, Farang Khuang
Majestic (สง่า): Khuen Phlapphla, Lao Somdet, Khaek Mon, Khaek Choen Chao, Sasom
Sacred (ศักดิ์สิทธิ์): Nang Nak, Tuang Phra That

Section 1: Hierarchy of Musical Logic
Priority (Highest→Lowest)
1. Thai Authenticity—tonal mode, Na Thap rhythmic cycle, phrasing, Luk Tok endings.
2. Rhythmic Structure—measure count and Na Thap length.
3. nWestern Notation Accuracy—strict ABC notation, correct meter, bar completeness, and note durations.
4. AI Procedure—validation, syntax formatting, header order.
When any rule conflicts, prioritize Thai authenticity while maintaining valid ABC syntax.

Section 2: Rhythmic Framework (หน้าทับ)—Adopt, do not choose
Use the same Na Thap type as the reference tune. The Na Thap fixes time signature, sectional structure, and total measures. Do not substitute the Na Thap and do not alter its standard length. Examples (reference only):
หน้าทับปรบไก่สองชั้น (Na Thap Prob Gai Song Chan): M:2/4; total 8 or 16 measures.
หน้าทับสองไม้สองชั้น (Na Thap Song Mai Song Chan): M:2/4; total 8 or 16 measures.
หน้าทับลาวสองชั้น (Na Thap Lao Song Chan): M:2/4; total 8 or 16 measures.
หน้าทับปรบไก่ชั้นเดียว (Na Thap Prob Gai Chan Diao): M:2/4; total 4 or 8 measures.
หน้าทับเขมรสองชั้น (Na Thap Khmer Song Chan): M:2/4; total 8 or 16 measures.
Rule:
- Total measures must equal the inherited Na Thap’s standard exactly (e.g., exactly 4, 8, 16, or 32 as applicable). 
- Arbitrary counts like 31 are invalid. No additional or incomplete measures.

Section 3 – Tempo Class (Adopted Attribute)
Adopt the same tempo class as the selected reference tune. Tempo class in Thai music depends on rhythmic rate (ชั้น / Chan): Sam Chan, Song Chan, or Chan Diao, which defines how quickly the Na Thap rhythmic cycle is performed.
Do not change or reinterpret the adopted tempo class.
Sam Chan (สามชั้น) Q:1/4 = 55–72 — slow rhythmic rate; dignified, lyrical pacing.
Song Chan (สองชั้น) Q:1/4 = 76–92 — medium rhythmic rate; balanced and flowing.
Chan Diao (ชั้นเดียว) Q:1/4 = 100–120 — fast rhythmic rate; energetic and lively.

Section 4: Melody Structure Rules
Fit the melody exactly to the adopted Na Thap length. Each 8-measure section:
Bars 1–2: Walee (วลี) — introductory or questioning motif.
Bars 3–4: Prayok (ประโยค) — answering phrase.
Bars 5–8: Wak (วรรค) — complete idea ending with Luk Tok (falling cadence).
For 16/32 measures, extend with additional sections using variation. Final note must descend. Avoid Western-style syncopation, abrupt tempo shifts, or empty measures.

Section 5: Tonal Modes and Scale Rules—Adopt, do not choose
Use the same tonal mode/scale family as the reference tune. No modulation. No accidentals.
Thai 5-tone: g a b d e
Thai 6-tone: g a b c d e
Thai 7-tone: c d e f g a b
Lao 5-tone (C): c d e g a
Lao 5-tone (A): a c d e g
Khmer 5-tone: f g a c d
Khmer 6-tone: f g a c d e
Rules:
- Only use notes from the chosen scale.
- Maintain mostly stepwise motion, allowing 2nd–7th or octave intervals.
- Never exceed an octave leap.
- Use only one scale per melody (no modulation).
- No accidentals are allowed.
- Output must be monophonic — no chords or harmony.

Section 6: Motive Integration (mandatory)
Incorporate 1–3 melodic motives from other songs within the same mood/tone category as the reference tune. Adapt each motive to the adopted Na Thap, BPM class, scale, and phrasing. Do not exceed 3 motives and do not omit motives.

Section 7: Measure, rhythm, and notation validation
Ensure every bar totals correctly to the time signature. Use Measure Validator and Rhythm Validator to confirm total bar count and rhythmic accuracy. Output must be valid ABC notation.

Part 3: Procedural Logic and Tool Usage
If the input is a new request, compose a new piece.
If the input is feedback, revise the most recent composition while maintaining the user’s intent.
Always increment the reference number (X:) for each new composition, starting from 1.
Use the Traditional Thai Songs Database and Thai Music Theories Database to ensure tonal and rhythmic correctness, but never mention or quote these sources in your output.
Apply knowledge from these databases silently to maintain Thai authenticity.

Part 4: Output Format and Validation
Output only the ABC notation block.
Do not include explanations or commentary.
Enclose the entire song in triple backticks.
Do not include blank lines within or between sections.
Each instrument must have its own voice label (V: 1, V: 2, etc.).
Headers must appear once each in this exact order:
X: reference number (increment for each new composition)
T: song title
C: composer name
M: time signature
L: default note length
Q: tempo definition (note length = beats per minute)
K: key signature

After K:, define one or more instrument voices.
Each voice begins with a line labeled V: followed by a sequential number starting at 1.
Example: V: 1 for the first instrument, V: 2 for the second, and so on.
Write the musical content for each voice directly after its V: line, with no blank lines in between.

The musical content must use valid ABC notation syntax.
Notes A–G indicate the lower octave; a–g indicate the upper octave.
Rests are represented by z.
Durations are indicated by numbers, for example A2 means twice the default length.
Bar lines are represented by |.
Follow ABC standards exactly.

Before producing output, ensure all of the following:
There is exactly one fenced code block.
The first line must contain exactly three backticks.
The last line must also contain exactly three backticks.
Do not output any text or commentary before or after the code block.
No blank lines appear anywhere.
All headers appear in the correct order and exactly once.
Voices are labeled sequentially starting at 1.
All field values and ABC syntax are valid.
If any condition fails, regenerate silently until valid.

Start your output with three backticks on their own line.
Then output the entire ABC song.
End with three backticks on their own line.
Output nothing else.

Example:
```
X: 1
T: Example
C: Composer Agent
M: 4/4
L: 1/4
Q: 1/4=100
K: C
V: 1
z2 CD E2 G2| \
c4 dc cc| \
cG Ac AG E2| \
G4 AG GG|
G2 EG EG Ac| \
A3c A2 G2| \
ED EG ED C2| \
D4 ED DD|
Dc AG c2 d2| \
e2 g4 c2| \
d2 eg de dc| \
A4 AG Ac|
ED CD E2 G2| \
A3c AG EG| \
AG Ac A2 GG| \
G4 AG GG|
G2 CD E2 G2|
```"""


example_persist_directory = os.path.join(path, "data", "examples")
example_collection_name = "examples"

theory_persist_directory = os.path.join(path, "data", "theories")
theory_collection_name = "theories"

example_vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=example_persist_directory,
        collection_name=example_collection_name
    )
example_retriever = example_vectorstore.as_retriever()
example_rag = create_retriever_tool(
    example_retriever,
    "example_rag",
    "Searches and returns traditional Thai songs in ABC notation, their motives, and their metadata from the database.",
)

theory_vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=theory_persist_directory,
        collection_name=theory_collection_name
    )
theory_retriever = theory_vectorstore.as_retriever()
theory_rag = create_retriever_tool(
    theory_retriever,
    "theory_rag",
    "Searches and returns the knowledge of traditional Thai music theories from the database.",
)


prompt_agent = create_react_agent(
    model=prompt_model,
    tools=[theory_rag],
    prompt=(prompt_system_prompt),
    name="prompt_agent",
)

composer_agent = create_react_agent(
    model=composer_model,
    tools=[example_rag, theory_rag],
    prompt=(composer_system_prompt),
    name="composer_agent",
)


def prompt_chat(user_input):

    for chunk in prompt_agent.stream({"messages": HumanMessage(content=user_input)}):
        for node, state in chunk.items():
            last_message = state["messages"][-1]

            if isinstance(last_message, AIMessage):
                if last_message.content:
                    print("\n==================== Prompt ====================")
                    if type(final_message.content) == list:
                        print(f"{last_message.content[0]['text']}\n")
                        return last_message.content[0]['text']
                    else:
                        print(f"{last_message.content}\n")
                        return last_message.content
                if getattr(last_message, "tool_calls", None):
                    for call in last_message.tool_calls:
                        print(f"[Tool Call] {call['name']} ({call.get('args')})")

            elif isinstance(last_message, ToolMessage):
                print(f"[Tool Response] {' '.join(last_message.content.split("\n")[0:3]).replace("\n", "")} ...")


def composer_chat(user_input):

    chat_history = st.session_state.chat_history
    chat_history.append(HumanMessage(content=user_input))

    final_message = None
    for chunk in composer_agent.stream({"messages": chat_history}):
        for node, state in chunk.items():
            last_message = state["messages"][-1]

            if isinstance(last_message, AIMessage):
                if getattr(last_message, "tool_calls", None):
                    for call in last_message.tool_calls:
                        print(f"[Tool Call] {call['name']} ({call.get('args')})")
                elif last_message.content:
                    final_message = last_message
                    print("\n==================== Result ===================")
                    if type(final_message.content) == list:
                        print(f"{final_message.content[0]['text']}\n")
                    else:
                        print(f"{final_message.content}\n")

            elif isinstance(last_message, ToolMessage):
                print(f"[Tool Response] {' '.join(last_message.content.split("\n")[0:3]).replace("\n", "")} ...")

    if final_message:
        chat_history.append(final_message)
        st.session_state.chat_history = chat_history

        if type(final_message.content) == list:
            song = final_message.content[0]['text']
        else:
            song = final_message.content
        song = song.split("```")[1]
        if song[0] == "\n": song = song[1:]
        if song[-1] == "\n": song = song[:-1]
        title = song.split("\n")[1].replace("T: ", "")
        song = song.replace("\n\n", "\n")
        
        if type(final_message.content) == list:
            return song, title
        else:
            return song, title


def abc_to_wav(song, song_name):

    temp_dir = tempfile.gettempdir()
    midi_path = os.path.join(temp_dir, f"{song_name}_{os.urandom(4).hex()}.mid")
    wav_path = os.path.join(temp_dir, f"{song_name}_{os.urandom(4).hex()}.wav")

    midi = Score.from_abc(song)
    midi.dump_midi(midi_path)

    fs = FluidSynth()
    fs.midi_to_audio(midi_path, wav_path)

    return midi_path, wav_path


st.set_page_config(page_title="Traditional Thai Melody Generation", page_icon="🎵")
st.title("Traditional Thai Melody Generation")

if "history" not in st.session_state:
    st.session_state.history = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "first_run" not in st.session_state:
    st.session_state.first_run = True

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"].replace("\n", "  \n"))
        if msg["role"] == "assistant":
            copy_button(msg["content"], tooltip="Copy", copied_label="Copied")

user_input = st.chat_input("Type or paste your prompt...")

if user_input:
    
    user_input = user_input.replace("\n", "  \n")
    st.chat_message("user").markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    with st.chat_message("assistant"):
        placeholder = st.empty()
        placeholder.markdown("_Composing..._")

        # if st.session_state.first_run:
        #     user_input = prompt_chat(user_input)
        #     st.session_state.first_run = False

        ai_response, song_name = composer_chat(user_input)
        midi_path, wav_path = abc_to_wav(ai_response, song_name)

        placeholder.audio(wav_path, format='audio/wav')

        with open(midi_path, "rb") as f:
            st.download_button("Download MIDI", f, song_name + ".mid", "audio/midi")
        
        ai_response = ai_response.replace("\n", "  \n")
        st.markdown("```" + ai_response)

        st.markdown("Click the **Copy** button below, then paste the text into https://notabc.app/abc-converter/ to view your sheet music.")

        copy_button(ai_response, tooltip="Copy", copied_label="Copied")

    st.session_state.history.append({"role": "assistant", "content": "```" + ai_response})