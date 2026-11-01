# AquaAlert — demo script (5–7 minutes)

A live walkthrough of the dashboard and API on the bundled sample data (Kerala floods, 15–16 August 2018).
Timings are a guide; the bold lines are what to say, the rest is what to do.

---

## Before you start (5 minutes, once)

| Step | Command / action | Why |
|---|---|---|
| 1 | `python scripts/fetch_docs.py` | The chatbot needs the NDMA / SDMA guideline documents in `data/docs/`. Without them it can only quote district briefs. |
| 2 | `python scripts/fetch_model.py` (already done if `models/muril/` exists) | MuRIL is the classifier the numbers below come from. |
| 3 | Check `GEMINI_API_KEY` in `.env` still has quota (one test question in the Chatbot tab). | If Gemini returns HTTP 429, briefs and the chatbot fall back to offline output with a warning: still demo-able, but say so. |
| 4 | Terminal 1: `streamlit run dashboard/app.py` → <http://localhost:8501> | |
| 5 | Terminal 2: `uvicorn api.main:app --reload` → <http://127.0.0.1:8000/docs> | |
| 6 | Load the dashboard once before presenting (first load classifies and geocodes 110 posts) and wait for the map tiles. | Avoids a spinner on stage. |
| 7 | Sidebar → **Replay time** → **Aug 15, 22:40**. Optional: **Pause background** on if you are screen-recording. | At this time the KPI row reads **105 posts · 17 rescue requests · 4 warnings · 4 low-credibility**. |

---

## 0:00 – 0:40 · The problem

**"During the 2018 Kerala floods, thousands of calls for help were posted on social media, in English, Malayalam,
Hindi, Bengali and in mixed, romanised forms like 'njangal veettil kudungi'. Responders had to read them by hand,
work out who needed rescue, where they were, which posts were rumours, and which offers of help matched which needs.
AquaAlert does that triage automatically and shows it on one screen."**

Point at the header: the replay chip (**Replay · 15 Aug 2018 22:40**), **7 languages**, **MuRIL**, **LLM: Gemini**.

## 0:40 – 1:30 · Architecture (one sentence per stage)

Show this (README → *Architecture*):

```
ingest → language → classify (+ distress rules) → extract → credibility ─┬→ summarize ─┐
                                                                          ├→ match       ├→ dashboard
                                                                          ├→ fuse ◄──────┘   api
                                                                          └→ chatbot (RAG)
```

**"Posts are replayed in time order like a live feed. We detect the language, including romanised and code-mixed
text. A fine-tuned MuRIL model sorts each post into seven categories, and a small rule layer catches clear distress
the model misses. We extract places, phone numbers and head counts, and geocode them, including Malayalam, Hindi and
Bengali place names. Every post gets a heuristic misinformation score. On top of that: district briefs, needs-to-offers
matching, early warnings that fuse post volume with rainfall and river levels, and a chatbot over NDMA guidelines.
Everything is also served by a FastAPI backend."**

## 1:30 – 2:30 · Map, a rule-fired rescue post, multilingual posts

**Map** tab. Point at the KPI cards, then the map: **"Each dot is a post, coloured by category; red is a rescue
request. The shaded circles are district risk levels."**

1. **Rule-fired rescue post — s045, Perumbavoor.** Click the red dot north-east of Kochi (Perumbavoor).
   Popup: *"Need rescue! 3 kids and 2 adults on the rooftop in Perumbavoor. Water is above 8 feet."* with the badges
   **rescue needed · rule: distress strong · no red flags** and the line *"the classifier said offer_help; a distress
   rule overrode it"*.
   **"MuRIL was trained on HumAID, where 'rescue' and 'boats' usually come from volunteers, so it read this as an
   offer. The rule layer sees an explicit 'need rescue' and promotes it, and records that it did."**
   (Other rule-fired posts if you want a second example: s021 Edathua "Rescue boats needed…", s020 Paravur "SOS!
   Pregnant woman needs evacuation".)
2. **Multilingual.** Click around Chengannur (south of Alappuzha):
   - **m014** (Malayalam script): *"ചെങ്ങന്നൂരിൽ ഞങ്ങളുടെ വീടിന്റെ മുകളിൽ 12 പേർ കുടുങ്ങി…"* → rescue needed, 85 %,
     placed in Chengannur from the Malayalam place name.
   - **m001** (Hindi): *"चेंगन्नूर में हमारे घर की छत पर 8 लोग फंसे हैं…"* → rescue needed.
   - **m009** (Hinglish): *"Hum Aluva me ghar ki chhat par phanse hain…"* → rescue needed via the contextual rule.
   **"Native-script posts transfer well from English training: Hindi 0.89, Bengali 0.86, Malayalam 0.73 accuracy on
   our hand-written set."**
3. Optional: the **Languages** filter in the sidebar to show only `ml`, then reset it.

## 2:30 – 3:15 · Credibility flags

Move **Replay time** to the end (**Aug 16, 08:00**). **"Overnight, a rumour spreads."** The low-credibility KPI rises
to **8**.

- Click the hollow, dashed marker at Mullaperiyar (**r001**): *"BREAKING: Mullaperiyar dam has broken!! … Forward to
  all groups!!!"* → **low credibility 0.75**, with the reasons: urges forwarding, dam-breach claim without a source,
  sensational punctuation, and **contradicted by an attributed update (r002: "Mullaperiyar dam is safe, there is no
  breach … District Collector, Idukki")**.
- The same rumour in Malayalam, **r013** *"മുല്ലപ്പെരിയാർ ഡാം പൊട്ടി! എല്ലാവർക്കും ഫോർവേഡ് ചെയ്യുക"*, scores 0.73.
- Contrast: the debunk **r007** (*"This photo is from the 2015 Chennai floods, not Aluva. Please don't forward it."*)
  scores **0** even though it mentions an old year.
- Tick **Hide low-credibility posts** in the sidebar: the dashed markers disappear everywhere, including briefs and
  matching. Untick it.

**"This is a transparent heuristic, not a trained model: it says *why* it is suspicious, and a score of zero means
'no known red flags', not 'verified'."**

Move **Replay time** back to **Aug 15, 22:40**.

## 3:15 – 3:50 · Daily brief

**Daily brief** tab → **Day: 2018-08-15** → **Generate briefs** (Gemini; cached after the first run).
Open **Alappuzha**: **"7 rescue requests, about 57 people, 2 missing-person reports"**, the most urgent posts first,
every claim cited with a post ID (s038 Kuttanad church, s001 Chengannur rooftop…), phone numbers redacted as
`<phone>` before the text goes to the LLM, and **"1 doubtful post left out"**.
The sources list under the summary links each cited post.

## 3:50 – 4:25 · Needs ↔ Offers

**Needs ↔ Offers** tab (defaults: text weight 0.60, max 50 km, one offer per need).

- **s021 → s022**: *"Rescue boats needed at Edathua, 30 people on a terrace"* ↔ *"Fishermen from Alappuzha are
  bringing their boats to help rescue operations"*, 21 km, score 0.62.
- **m010 → s035**: Hinglish *"Kochi relief camp me dawai aur khana chahiye"* ↔ English *"Doctors available for free
  medical aid at Kochi relief camps"*, score 0.71. **"Matching works across languages because the embeddings are
  multilingual."**
- Drag **Text weight** down to show proximity taking over.

## 4:25 – 5:00 · Warnings

**Warnings** tab (15 Aug): **4 WARNING, 8 WATCH**.
Open **Alappuzha**: *"9 rescue/damage posts vs 0.0/day over the previous 3 days | rainfall 196 mm (very heavy;
threshold 115.6 mm)"*. **"A warning needs both: people posting for help *and* a sensor threshold."** Idukki also shows
*"Periyar at 91 % of its danger level"*. Pick **Idukki** in **Sensor history** to show the rainfall peak and the river
rising a day later. On the **Map**, the district risk layer shows the same levels.
**"The sensor files here are synthetic but shaped like August 2018; real IMD and CWC sources are in the README."**

## 5:00 – 5:45 · Chatbot in Hindi

**Chatbot** tab. Ask: **`बाढ़ से पहले क्या तैयारी करनी चाहिए?`** ("How should we prepare before a flood?")
The answer comes back in Hindi from the citizen guidance (NDMA *Floods: Do's & Don'ts*, the Manipur SDMA sheet),
with numbered citations. Open **Sources**: each citation shows the document, page and a *citizen guidance* /
*policy document* / *district brief* tag.
Optional follow-up: **"What is the role of the state government in flood forecasting?"** → answered from the NDMA
policy guidelines instead. **"Personal-safety questions are steered to household guidance, government questions to
policy documents, and the bot tells you to call 112 if you are in danger."**

## 5:45 – 6:15 · API docs

Open <http://127.0.0.1:8000/docs>.
- `GET /warnings` → **Try it out** → `day=2018-08-15`, `level=warning` → the same four districts with reasons.
- `POST /classify` with `{"text": "Hum Aluva me ghar ki chhat par phanse hain, 6 log hain, jaldi boat bhejo"}` →
  `label: rescue_needed`, `rule: distress_contextual`, `model_label: offer_help`, `language: hi-Latn`,
  district **Ernakulam**, `people_count: 6`, misinformation score 0.
**"Everything on the dashboard is available to other tools: posts with filters, matches, briefs, warnings, chat."**

## 6:15 – 7:00 · Results and honest limitations

**HumAID test split, 15,160 tweets** (macro-F1 with 95 % bootstrap CI):

| Model | Macro-F1 (all 7) | Macro-F1 (5 large classes) | Accuracy |
|---|---:|---:|---:|
| Keyword fallback | 0.315 (0.302–0.327) | 0.342 | 0.354 |
| embed_lr (MiniLM + logistic regression, Mac) | 0.605 (0.582–0.625) | 0.687 | 0.734 |
| **MuRIL fine-tuned (Colab T4)** | **0.727 (0.700–0.752)** | **0.759** | **0.800** |

Per language (accuracy on 44 hand-written multilingual posts + the English sample; tiny, synthetic sets):

| | Hindi | Bengali | Malayalam | Hinglish | Manglish | Banglish |
|---|---:|---:|---:|---:|---:|---:|
| MuRIL | 0.89 | 0.86 | 0.73 | 0.44 (0.56 with rules) | 0.43 | 0.17 |

**"Native script works; romanised, code-mixed text is the weak spot. A retrained model with 4,552 LLM-generated
romanised posts is training on Kaggle now."**

Limitations to say out loud:
- **Romanised text is weak** (0.17–0.56 above). Examples visible on the map: **s028** *"…6 jon manush bondi, nouka
  dorkar"* (Banglish rescue) shows as *need supplies*; **m038** (Banglish rooftop SOS) as *irrelevant*.
- **The rescue class is tiny in HumAID** (22 test tweets), so its F1 and the distress-rule numbers are noisy; on
  HumAID the rule layer slightly lowers macro-F1 (0.727 → 0.716) while fixing the sample's missed SOS posts.
- **Credibility is heuristic**, tuned on 16 synthetic rumour posts; treat scores as "check this", never as proof.
- **Synthetic demo data**: posts, rumours and sensor readings are made up (shaped like 2018); the romanised
  training data is LLM-generated (~10 % meaning drift in a spot check; Bengali short of target).
- **Geocoding** is a gazetteer of ~35 Kerala places plus optional Nominatim; unknown villages are not mapped.
- **Matching inherits classifier errors** (see the issues list below).
- The API has **no authentication**; run it locally.

**"AquaAlert is a working end-to-end prototype: every stage is testable on its own, everything runs offline with
fallbacks, and the next step is better romanised-language data."**

---

## Dashboard issues noticed while preparing this script

Found while checking each feature on the sample data (15–16 Aug replay, MuRIL, Gemini):

1. **Romanised / code-mixed posts misclassified on the map.** s028 (Banglish rescue) → *need supplies* (49 %);
   m038 (Banglish rooftop SOS) → *irrelevant* (89 %); m024 (Manglish) and m043 (Malayalam + English) rescue posts →
   *need supplies*. Expected given the per-language results; avoid clicking these unless you are making that point.
2. **Matches inherit those errors.** m037 (Bengali *"…please rescue us"*) is classified as an *offer* and gets matched
   to m031, another stranded group; m027 (Manglish offer of a vehicle) is treated as a *need*; s041 (volunteers to
   clean houses) is matched to a food-and-water request (s005) mainly because both are in Pandalam. Show s021→s022 and
   m010→s035 instead.
3. **Avoid ad-lib Hinglish in `/classify`.** *"Hum chhat par phase hue hain, please help! Aluva"* comes back as
   *need supplies* with no rule (rules only override *offer help* / *info update*); use the m009 sentence above.
4. **Default replay time is 16 Aug 08:00**, when no district is at WARNING (0 warnings, 13 watch); move the slider to
   15 Aug 22:40 for the warnings segment.
5. **The Daily brief tab defaults to the latest day (16 Aug)**, which has few posts; pick 2018-08-15. With Gemini
   configured, briefs only appear after **Generate briefs**.
6. **Chatbot without `fetch_docs.py`** shows a yellow notice and answers only from district briefs (the Hindi question
   then gets "no information in the passages"). Run step 1 of the prep.
7. **Gemini quota**: an HTTP 429 earlier in the day means briefs and chatbot fall back to offline output with a notice.
8. **Map tiles load slowly on first view** (grey squares for a few seconds); load the dashboard before presenting.
9. **Fixed while preparing** (in this commit): r010 *"Water has receded … no more rescue needed"* was promoted to a
   rescue request by the distress rules ("rescue needed" inside "no more rescue needed"); briefs swallowed the
   `<phone>` redaction marker as an HTML tag; the posts table crashed when blank and numeric head counts were mixed.
10. **Not fully themed**: the posts table is canvas-drawn (Glide Data Grid), so it takes the theme colours from
   `.streamlit/config.toml` but not the Montserrat uppercase headers or navy header rule; the rainfall chart's axis
   fonts and the OpenStreetMap tiles keep their own styling.
