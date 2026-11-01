> **Presenter notes (English; not part of the pitch).** This is the Hinglish version of
> [docs/demo_script.md](demo_script.md); do the same prep (fetch docs, check Gemini, Replay at **15 Aug 22:40**) and
> show the same sample posts. Post IDs per step:
>
> | Step | Show | English script section |
> |---|---|---|
> | 30-second pitch / 1. Problem | Header + KPI row (105 posts · 24 rescue requests · 4 warnings · 4 low-credibility) | 0:00 The problem |
> | 2. Classifier | **m014** (Malayalam), **m001** (Hindi): both *rescue needed* | 1:30 Map |
> | 3. Rule layer | **s045** Perumbavoor ("rule: distress strong", classifier said *need_supplies*); **m009** Hinglish "…jaldi boat bhejo"; **s020** for the *confirms* badge | 1:30 Map |
> | 4. Location | **m014**: placed in Chengannur from the Malayalam place name; **s045** Perumbavoor | 1:30 Map |
> | 5. Credibility | Replay → **16 Aug 08:00**: **r001** (dam rumour, 0.75, contradicted by **r002**), **r013** (Malayalam, 0.73), debunk **r007** (0); then back to 15 Aug 22:40 | 2:30 Credibility |
> | 6. Daily brief | Day **2018-08-15** → **Alappuzha** (cites **s038**, **s001**) | 3:15 Daily brief |
> | 7. Matching | **s021 → s022** (boats), **m010 → s035** (Hinglish need ↔ English offer). Avoid m011, m019, s038→r012 | 3:50 Needs ↔ Offers |
> | 8. Early warning | Warnings tab, 15 Aug: **Alappuzha** (13 posts, 196 mm), **Idukki** (Periyar at 91 %) | 4:25 Warnings |
> | 9. Chatbot | Ask **"बाढ़ से पहले क्या तैयारी करनी चाहिए?"** (needs `scripts/fetch_docs.py`) | 5:00 Chatbot |
> | 10. Live mode | Dataset → LIVE: official SACHET alerts in the Warnings tab; **Simulate incident** (8 SIMULATED posts, Hinglish SOS first), then **Clear** | Pre-demo checklist |
> | Results | Results tables in the English script / README | 6:15 Results |
>
> Differences from the English script and README (kept as written below, flagged here):
> - Step 4 says "Chengannur ya Kanpur": only ~35 **Kerala** places are mapped; Kanpur would not get a pin.
> - The rule-layer limitation quotes 0.727 → 0.722 (English-only MuRIL). The demo runs the romanised model, where it is
>   0.736 → 0.731 (as in the English script).
> - "News aur Bluesky polling bani hai": neither was verified live (GDELT returned HTTP 429 to this machine; no Bluesky
>   credentials configured), as the README says.
> - Step 7 mentions only proximity; the match score is 60 % text similarity + 40 % proximity.
> - The text received ended mid-question ("Accuracy…"); the rest was not received.

# AquaAlert: Hinglish pitch

## 30-second pitch
Jab bhi koi badi baadh aati hai, jaise Kerala 2018, social media pe hazaaron posts aate hain: koi bol raha hai "hum chhat pe fase hain", koi khana de raha hai, koi galat afwah phaila raha hai. Itne saare posts manually padhna impossible hai. AquaAlert ek AI system hai jo ye posts khud padhta hai, samajhta hai ki kaun si emergency hai, kahaan hai, kitni reliable hai, aur sab kuch ek map pe dikhata hai. Ye Hindi, Malayalam, Bengali aur Hinglish jaisi Indian languages mein kaam karta hai.

## 5-minute walkthrough
1. Problem: Baadh ke time pe rescue teams ke paas time kam hota hai. Posts alag-alag languages mein hote hain, aur kaafi fake news bhi hoti hai. Isliye humne ek pipeline banayi jo text ko actionable information mein badalti hai.
2. Classifier: Har post ko 7 categories mein baantta hai: rescue needed, supplies chahiye, help offer, missing person, infrastructure damage, info update, irrelevant. Humne MuRIL model fine-tune kiya, jo Indian languages ke liye bana hai.
3. Rule layer: Agar model kisi clear SOS ko galat label de, jaise "bachao, chhat pe fase hain", toh ek safety rule use rescue request bana deta hai. Dashboard dikhata hai ki rule kab chala.
4. Location: Post se jagah ka naam nikaalta hai, jaise Chengannur ya Kanpur, aur map pe pin lagata hai.
5. Credibility: Rumours pakadta hai, jaise "forward to all" wali posts, ya dam toot gaya wali afwah jo officially deny ho chuki hai. Ye rule-based hai, trained model nahi.
6. Daily brief: Gemini AI se district-wise summary banta hai, har claim ke saath post ID cite hoti hai.
7. Matching: Jinko help chahiye aur jo help de rahe hain, unhe paas ke hisaab se match karta hai.
8. Early warning: Posts ki sankhya aur barish ya river level dono dekh ke district ko WATCH ya WARNING deta hai.
9. Chatbot: NDMA ki guidelines pe based chatbot, jo Hindi mein bhi jawab deta hai aur source batata hai.
10. Live mode: Official NDMA SACHET alerts live aate hain. Jab asli baadh nahi hoti, feed shaant rehta hai, isliye Simulate incident button hai, jo SIMULATED label ke saath demo posts daalta hai.

## Results
Keyword approach ka macro-F1 0.315 tha. Embeddings se 0.605. Fine-tuned MuRIL se 0.727, aur romanised Hinglish/Manglish data ke saath 0.736. Test 15,000 se zyada real tweets pe hua.

## Honest limitations
- Demo ke posts aur sensor data synthetic hain, kyunki live flood data tab hi milta hai jab asli baadh ho. Models aur NDMA documents asli hain.
- Live mode mein SACHET alerts verified hain. News aur Bluesky polling bani hai, lekin unki availability feed pe depend karti hai.
- Romanised training data AI se banaya gaya hai, isliye real social media pe accuracy kam ho sakti hai. Per-language test sets bahut chhote hain (6-11 posts).
- Rescue class ke sirf 22 test examples the, isliye us number pe zyada bharosa nahi karna chahiye.
- Rule layer HumAID score ko thoda kam karta hai (0.727 se 0.722), lekin rescue requests miss na hon, isliye rakha hai.

## Possible questions
- "Ye real-time hai?" Official SACHET alerts live hain. Baaki feeds quiet reh sakte hain jab baadh nahi hoti, isliye replay mode aur simulate button bhi hain.
- "Accuracy

<!-- The pitch text received ended here, mid-question. -->
