# Transcript Analysis Dimensions

Comprehensive reference of every dimension extracted from interview transcripts for LLM-assisted response detection. Each dimension is classified by extraction method (Programmatic / LLM) and detection layer (NLP Layer / Semantic Layer).

---

## Programmatic Dimensions (NLP Layer)

These are computed from text statistics using established NLP libraries. No subjective judgment — pure measurement.

---

### 1. Type-Token Ratio (TTR)

**What it is:** The ratio of unique words to total words in a turn. Measures vocabulary diversity.

**How it works:** If a speaker uses 100 words but only 60 are unique, TTR = 0.60. Higher = more diverse vocabulary. Lower = more repetitive.

**How it helps:** LLM-generated text tends to use a narrower, more formulaic vocabulary (repeating phrases like "I believe," "at the same time," "overall"). Genuine speech has more accidental variety. When a speaker's TTR drops significantly below their own baseline in a long answer, it suggests they're reading from a source with repetitive phrasing.

**Extraction:** Programmatic — `nltk.word_tokenize` + set cardinality.

**Library:** `nltk`

---

### 2. Hapax Legomena Ratio

**What it is:** The proportion of words that appear exactly once in a turn.

**How it works:** Count words that occur only one time, divide by total unique words. High ratio = many one-off word choices. Low ratio = words are being reused.

**How it helps:** Genuine speakers use many words once (natural variation). LLM text tends to reuse the same phrases. A low hapax ratio relative to the speaker's baseline suggests templated or generated text.

**Extraction:** Programmatic — `nltk.FreqDist` on tokenized words.

**Library:** `nltk`

---

### 3. Average Sentence Length

**What it is:** Mean number of words per sentence in a turn.

**How it works:** Split text into sentences, count words in each, compute the average.

**How it helps:** LLMs produce consistently long, grammatically complete sentences. Genuine speakers mix short fragments with longer sentences. An unusually high average sentence length (well above the speaker's baseline) suggests the text was generated rather than spoken naturally.

**Extraction:** Programmatic — `nltk.sent_tokenize` + `nltk.word_tokenize`.

**Library:** `nltk`

---

### 4. Sentence Length Variance

**What it is:** How much sentence length varies within a single turn. Measures structural uniformity.

**How it works:** Compute the variance of sentence lengths (in words) across all sentences in a turn. High variance = mix of short and long sentences (natural). Low variance = all sentences are similar length (suspicious).

**How it helps:** This is one of the strongest signals. LLM-generated text has suspiciously uniform sentence lengths — every sentence is roughly 15-25 words. Genuine speech has high variance: some 3-word fragments, some 30-word run-ons. When variance drops significantly below a speaker's median, it flags reading from a generated source.

**Extraction:** Programmatic — sentence tokenization + variance computation.

**Library:** `nltk`

---

### 5. Function Word Ratio

**What it is:** The proportion of function words (articles, prepositions, conjunctions, pronouns) vs. content words.

**How it works:** Count words that appear in the NLTK English stopwords list, divide by total words.

**How it helps:** Function words ("the", "and", "in", "it") are used unconsciously in natural speech. LLM-generated text tends to have a lower function word ratio because it uses more formal, content-heavy language. A significant drop below baseline suggests the text was crafted rather than spoken.

**Extraction:** Programmatic — stopword lookup via `nltk.corpus.stopwords`.

**Library:** `nltk`

---

### 6. Long Word Ratio

**What it is:** The proportion of words longer than 6 characters.

**How it works:** Count words with >6 characters, divide by total words.

**How it helps:** Longer words correlate with formal/academic vocabulary. LLM answers tend to use more long words ("authentication", "implementation", "collaboration") than a speaker naturally would. When this ratio spikes above the speaker's baseline, it signals elevated vocabulary from an external source.

**Extraction:** Programmatic — character length counting.

**Library:** Standard Python

---

### 7. Flesch Reading Ease

**What it is:** A readability score (0-100) measuring how easy text is to read. Higher = easier.

**How it works:** Combines sentence length and syllable count into a single score. 100 = very easy (children's books). 30 = very difficult (academic papers).

**How it helps:** Establishes the complexity baseline for each speaker. LLM answers tend to score lower (harder to read) than a speaker's natural register. A large gap between casual turns and substantive answers indicates a register shift.

**Extraction:** Programmatic — `textstat.flesch_reading_ease`.

**Library:** `textstat`

---

### 8. Flesch-Kincaid Grade Level

**What it is:** A readability score that maps text to a U.S. school grade level. Grade 8 = readable by an average 8th grader.

**How it works:** Combines sentence length and syllables-per-word into a grade number.

**How it helps:** Same as Flesch Reading Ease but more interpretable. A speaker whose casual English is at grade 6 but whose answers jump to grade 14 has undergone a register shift. The gap (FK grade gap) is a direct measure of how much the answer's complexity differs from the speaker's natural level.

**Extraction:** Programmatic — `textstat.flesch_kincaid_grade`.

**Library:** `textstat`

---

### 9. Average Syllables Per Word

**What it is:** Mean syllable count across all words in a turn.

**How it works:** Count syllables in each word, compute the average.

**How it helps:** More syllables per word = more complex vocabulary. Tracks with the long word ratio but captures a different dimension (polysyllabic common words like "understanding" vs. monosyllabic rare words). Used as a feature in the outlier detection model.

**Extraction:** Programmatic — `textstat.avg_syllables_per_word`.

**Library:** `textstat`

---

### 10. Proselint Issue Count

**What it is:** Number of grammar/style issues detected by proselint.

**How it works:** Runs a comprehensive grammar checker that catches repeated words, clichés, weasel words, lexical illusions, and other style issues.

**How it helps:** Genuine speakers make consistent types of errors (repeated words from stuttering, informal constructions). LLM text is usually error-free. A sudden absence of proselint issues in a normally error-prone speaker is suspicious. Conversely, a spike in issues might indicate the speaker is struggling (genuine).

**Extraction:** Programmatic — `proselint.tools.LintFile`.

**Library:** `proselint`

---

### 11. POS-Based Formality Score

**What it is:** A 0-1 score measuring how formal the grammatical structure of the text is, based on the ratio of formal parts-of-speech (nouns, adjectives, prepositions, determiners) to informal ones (pronouns, verbs, adverbs, interjections).

**How it works:** Tags every word with its part of speech using spacy. Counts formal POS tags vs. informal POS tags. Based on the Heylighen & Dewaele (2002) formality metric.

**How it helps:** This is the core of register detection. Formal/LLM text is noun-heavy ("The implementation of the authentication system requires careful consideration"). Casual/genuine speech is verb/pronoun-heavy ("I implemented the auth system and it was tricky"). The formality gap between a speaker's casual turns and their substantive answers is a direct measure of register shift — the strongest signal for hybrid LLM users.

**Extraction:** Programmatic — `spacy` POS tagging + ratio computation.

**Library:** `spacy`

---

### 12. Disfluency Ratio

**What it is:** The ratio of interjections (filler words, hesitations) to total words.

**How it works:** Spacy tags interjections (INTJ POS tag) — words like "uh", "um", "hmm", "well". Count them, divide by total words.

**How it helps:** When you're thinking on your feet, you naturally produce fillers. When you're reading from a screen, you don't. A speaker whose casual turns have normal disfluency but whose long answers have zero disfluency is reading pre-written text. This is particularly powerful for non-native English speakers who naturally hesitate more.

**Extraction:** Programmatic — `spacy` POS tagging, count INTJ tags.

**Library:** `spacy`

---

### 13. Average Dependency Parse Depth

**What it is:** Mean depth of the syntactic dependency tree. Measures sentence complexity at the grammatical level.

**How it works:** For each word, count how many "hops" up the dependency tree to reach the root. Average across all words.

**How it helps:** LLM text tends to have deeper, more nested syntactic structures ("The system, which was designed to handle authentication flows that involve multiple middleware layers, processes requests efficiently"). Genuine speech has flatter structures. A spike in parse depth above baseline suggests generated text.

**Extraction:** Programmatic — `spacy` dependency parsing.

**Library:** `spacy`

---

### 14. Noun Ratio / Verb Ratio

**What it is:** Proportion of nouns and verbs in the text.

**How it works:** Count words tagged as NOUN or VERB by spacy, divide by total words.

**How it helps:** LLM text is noun-heavy (descriptive, abstract). Genuine speech is verb-heavy (action-oriented, personal). These ratios feed into the formality score but also serve as independent features in the outlier detection model.

**Extraction:** Programmatic — `spacy` POS tagging.

**Library:** `spacy`

---

### 15. Average Zipf Frequency (Lexical Sophistication)

**What it is:** The average frequency score of words used, on the Zipf scale (1 = very rare, 7 = extremely common).

**How it works:** Look up each word's frequency in a large English corpus. Average the scores. Lower average = rarer, more sophisticated vocabulary.

**How it helps:** LLMs tend to use rarer, more academic words than real people naturally would. A speaker whose average Zipf score drops significantly below their baseline (i.e., they're suddenly using rarer words) is likely pulling vocabulary from an external source. This is more robust than a hardcoded "abstract nouns" list because it captures ANY rare word, not just known ones.

**Extraction:** Programmatic — `wordfreq.zipf_frequency` (corpus-backed, no word lists).

**Library:** `wordfreq`

---

### 16. Rare Word Ratio

**What it is:** Proportion of words with Zipf frequency below 4.0 (relatively rare in everyday English).

**How it works:** Count words with Zipf < 4.0, divide by total words.

**How it helps:** Complements the average Zipf score. Even if the average is close to baseline, a high rare-word ratio might indicate several unusually sophisticated words dropped into otherwise normal text — a pattern seen when someone edits LLM output slightly.

**Extraction:** Programmatic — `wordfreq.zipf_frequency` + threshold counting.

**Library:** `wordfreq`

---

### 17. GPT-2 Perplexity

**What it is:** How "surprised" a language model (GPT-2) is by the text. Lower perplexity = more predictable = more likely machine-generated.

**How it works:** Feed the text into GPT-2, compute the loss (how well GPT-2 could have predicted each word), exponentiate to get perplexity. GPT-2 perplexity of 20 = very predictable (LLM-like). GPT-2 perplexity of 100+ = surprising/varied (human-like).

**How it helps:** This is the strongest standalone text signal. LLMs generate text by picking the most probable next token — so LLM output has inherently low perplexity when scored by another LM. Genuine human speech has higher perplexity because people make unexpected word choices, use regionalisms, and construct unusual sentences. When a turn's perplexity is significantly below the speaker's own median, it flags reading from a generated source.

**Extraction:** Programmatic — `transformers` (GPT-2 model), lazy-loaded.

**Library:** `transformers` + `torch`

---

### 18. TF-IDF Structural Uniformity (Template Score)

**What it is:** A 0-1 score measuring how similar all of a speaker's answers are to each other in word patterns.

**How it works:** Convert each answer into a TF-IDF vector (term frequency–inverse document frequency). Compute pairwise cosine similarity between all answer vectors. Average the similarities. High score = answers are very similar (template-like). Low score = answers are varied (natural).

**How it helps:** Template answers — whether from memorization, pre-written scripts, or LLM generation — share structural patterns even when the specific content differs. "First, I would... Another thing I would... Overall, I believe..." produces similar TF-IDF vectors across turns. Genuine speakers vary their structure naturally. A high template score (above ~0.15) suggests answers follow a repeated pattern.

**Extraction:** Programmatic — `sklearn.feature_extraction.text.TfidfVectorizer` + `sklearn.metrics.pairwise.cosine_similarity`.

**Library:** `sklearn`

---

### 19. Register Gap

**What it is:** The difference in formality, vocabulary rarity, and readability between a speaker's short/conversational turns and their long/substantive answers.

**How it works:** Split turns into "conversational" (<15 words) and "substantive" (>60 words). Compute median formality score, median Zipf score, and median FK grade for each group. The gap between them is the register shift.

**How it helps:** This is the key signal for hybrid LLM users (like Priyanka). Her casual English was intermediate-level, but her technical answers were textbook-perfect. The gap proves the answers came from a different source than her natural voice. Three sub-metrics:
- **Formality gap:** POS-based formality difference
- **Zipf gap:** Vocabulary rarity difference
- **FK grade gap:** Readability level difference

**Extraction:** Programmatic — computed from the above features, grouped by turn length.

**Library:** Uses spacy + wordfreq + textstat outputs

---

### 20. Isolation Forest Outlier Score

**What it is:** A continuous score indicating how much a turn is a multivariate outlier relative to the speaker's other turns.

**How it works:** Takes all 11 features above as a feature vector per turn. Isolation Forest (an unsupervised anomaly detection algorithm) learns the speaker's "normal" distribution and scores each turn. Negative score = outlier. More negative = more anomalous.

**How it helps:** Catches turns that are *jointly* abnormal across multiple dimensions — something per-feature thresholds miss. A turn might be only slightly unusual on formality, vocabulary, AND perplexity individually, but the combination makes it an outlier. No hardcoded thresholds — the model adapts to each speaker's natural variance.

**Extraction:** Programmatic — `sklearn.ensemble.IsolationForest`.

**Library:** `sklearn`

---

### 21. Z-Scores (Per Feature, Per Turn)

**What it is:** How many standard deviations each feature is from the speaker's own mean. A z-score of +2 means the value is 2 standard deviations above the speaker's average.

**How it works:** For each feature, compute mean and standard deviation across all of the speaker's turns. Then for each turn, compute (value - mean) / std.

**How it helps:** Provides fine-grained, per-turn evidence of deviation. Instead of just "this turn is suspicious," it tells you *why*: "formality z=+2.1 (abnormally formal), disfluency z=-1.8 (abnormally few fillers), zipf z=-1.5 (abnormally rare vocabulary)." These feed into the composite score and are also passed to the LLM layer as evidence.

**Extraction:** Programmatic — numpy mean/std + z-score computation.

**Library:** `numpy`

---

### 22. Composite Score

**What it is:** A single 0-1 suspicion score per turn, combining all programmatic signals.

**How it works:** Weighted sum of z-score contributions from each feature, register gap, template score, and isolation forest outlier flag. Weights are configurable and can be calibrated with labeled data.

**How it helps:** Triage tool for deciding whether to run the expensive LLM layer:
- **< 0.3:** Likely genuine — skip LLM layer (saves cost)
- **0.3–0.5:** Ambiguous — run LLM layer for confirmation
- **> 0.5:** Likely LLM — run LLM layer for evidence and detailed analysis

**Extraction:** Programmatic — weighted combination of all above features.

---

## LLM-Based Dimensions (Semantic Layer)

These require an LLM to evaluate because they involve understanding meaning, context, and reasoning about the speaker.

---

### 23. Speaker Profile

**What it is:** A structured assessment of the speaker's natural English ability, built from their conversational/short turns.

**What the LLM evaluates:**
- English proficiency level (beginner → native)
- Natural register (very casual → very formal)
- Common grammatical patterns and errors
- Vocabulary level (basic → academic)
- Discourse markers they naturally use (so, like, yeah, actually...)
- Sentence structure habits (fragments? run-ons? complex?)
- Topics they demonstrate genuine understanding of
- How they express uncertainty (asks for clarification? hedges? trails off?)
- Likely native language

**How it helps:** Creates a rich baseline that captures what statistics miss. The LLM can distinguish between someone who *chooses* simple words (fluent but casual) vs. someone who *only knows* simple words (limited proficiency). It can also identify the speaker's natural register shift patterns, topic knowledge, and error tendencies.

**Extraction:** LLM — analyzed from conversational turns and short answers.

---

### 24. Register Match (Per Answer)

**What it is:** A 1-10 score rating how well an answer's formality and quality match the speaker's natural register.

**What the LLM evaluates:** Does the English quality of this specific answer match what the speaker demonstrated in their conversational turns? Is it too polished? Too formal? Too structured?

**How it helps:** Catches the Priyanka pattern — answers that are grammatically perfect when the speaker's baseline has "um"s and restarts. The LLM can detect register shifts that are too subtle for POS-based formality scoring.

**Extraction:** LLM — evaluated per answer against the speaker profile.

---

### 25. Vocabulary Match (Per Answer)

**What it is:** A 1-10 score rating whether the words used in the answer match the speaker's demonstrated vocabulary level.

**What the LLM evaluates:** Are there words in this answer that the speaker wouldn't naturally use? "Multifaceted," "measurable deliverables," "short progress check point" — would this speaker actually use these words?

**How it helps:** Catches vocabulary injection from LLM or pre-preparation. Zipf frequency catches rare words statistically, but the LLM can reason about *context* — "measurable deliverables" might not be a rare phrase, but it's unusual for a college student with no work experience.

**Extraction:** LLM — evaluated per answer against the speaker profile.

---

### 26. Likely Origin (Per Answer)

**What it is:** A classification of where the answer likely came from.

**Categories:**
- `real_time` — thought up on the spot (genuine)
- `recalled_from_memory` — a real experience being narrated from recall
- `pre_written_script` — reading from prepared material (could be notes or LLM-generated)
- `llm_generated` — reading LLM output in real-time or from a prepared script

**What the LLM evaluates:** Does the answer show signs of real-time composition (hesitations, evolving structure, self-corrections)? Or does it read like pre-formed text (uniform quality, complete sentences, no restarts)?

**How it helps:** Distinguishes between thinking on your feet vs. reading from a screen. Statistics can approximate this (disfluency ratio, sentence variance), but the LLM can make nuanced judgments about the *texture* of the speech.

**Extraction:** LLM — evaluated per answer.

---

### 27. Specificity (Per Answer)

**What it is:** Whether the answer contains concrete personal details or is generic/abstract.

**Categories:** `high` | `medium` | `low` | `none`

**What the LLM evaluates:** Does the speaker mention specific project names, numbers, dates, tools, or personal experiences? Or is the answer generic advice that could apply to anyone?

**How it helps:** Genuine speakers give specific, often imprecise details ("we had like 1.4 lakhs of records," "it took two and a half weeks"). LLM answers tend to be generic ("collaboration is important," "I believe in continuous learning"). Low specificity in a long answer is suspicious.

**Extraction:** LLM — evaluated per answer.

---

### 28. LLM Markers Found (Per Answer)

**What it is:** A list of specific LLM-typical phrases or patterns found in the answer.

**What the LLM evaluates:** Does the answer contain phrases like "That's a really good question," "It's worth noting," "It's pretty straightforward"? Does it use balanced "on one hand / on the other hand" structure? Does it have textbook definition patterns ("X is a Y that does Z")?

**How it helps:** Direct evidence of LLM generation. The LLM can recognize its own output patterns — cliché phrases, structural templates, hedging patterns — that are hard to capture with regex or word lists because they vary in form.

**Extraction:** LLM — evaluated per answer.

---

### 29. Structural Pattern (Per Answer)

**What it is:** How the answer is organized.

**Categories:** `narrative` (telling a story) | `list` (enumerating points) | `template` (filling a structure) | `textbook` (defining/explaining like a textbook) | `rambling` (stream of consciousness)

**What the LLM evaluates:** Does the answer follow a recognizable pattern? "First... Another thing... Also... Overall..." is a template. "So what happened was, I was working on..." is narrative.

**How it helps:** Template and textbook patterns suggest LLM or heavy preparation. Narrative and rambling patterns suggest genuine thought. The LLM can distinguish these even when the specific words change.

**Extraction:** LLM — evaluated per answer.

---

### 30. Confidence Contradiction (Per Answer)

**What it is:** Whether the speaker's confidence in this answer contradicts their demonstrated knowledge elsewhere in the session.

**What the LLM evaluates:** Does the speaker confidently explain AWS vs. Azure cloud architecture but then struggle with basic testing concepts? If the knowledge level is inconsistent across topics, some answers may have been externally sourced.

**How it helps:** Catches selective LLM use. A speaker who uses LLM for hard questions but answers easy ones genuinely will show knowledge gaps that the LLM can detect. Statistics can't reason about topic difficulty — the semantic layer can.

**Extraction:** LLM — cross-referencing across answers.

---

### 31. Cross-Answer Consistency Score

**What it is:** A 0-1 score measuring how consistent all answers are with each other in register, vocabulary, and knowledge level.

**What the LLM evaluates:** Are all answers at the same quality level? Or are some dramatically better than others? Is the vocabulary level stable? Is the demonstrated knowledge consistent?

**How it helps:** Genuine speakers are consistently imperfect. LLM users have some answers that are suspiciously perfect mixed with natural ones. The pattern of inconsistency (which answers are different) reveals the LLM usage pattern.

**Extraction:** LLM — analyzed across all answers together.

---

### 32. Pattern Description

**What it is:** A natural language description of the observed pattern across answers.

**Examples:**
- "Concept-definition answers are textbook-perfect while personal answers are rough"
- "First answer is highly polished, subsequent answers degrade"
- "All answers uniformly polished — no genuine baseline visible"
- "All answers rough but coherent — genuine throughout"

**How it helps:** Provides human-readable evidence for the verdict. Instead of just "score 0.7," the system can say "Turns 4, 6, 8, 10 are LLM-generated (concept explanations), while turns 14, 16, 18 are genuine (personal stories)."

**Extraction:** LLM — synthesized from all analyses.

---

### 33. Overall Assessment

**What it is:** The session-level verdict.

**Categories:**
- `genuine` — all answers from the speaker's own knowledge
- `llm_primary` — most answers are LLM-assisted
- `mixed_genuine_and_llm` — some answers LLM, some genuine
- `pre_prepared_with_llm` — LLM was used before the interview to prepare, answers delivered from memory
- `insufficient_data` — not enough turns to assess

**How it helps:** The final product. Combines all programmatic and LLM evidence into a single actionable verdict with a confidence score.

**Extraction:** LLM — based on all evidence from both layers.

---

## Summary Table

| # | Dimension | Method | Library | Layer |
|---|---|---|---|---|
| 1 | Type-Token Ratio | Programmatic | nltk | NLP |
| 2 | Hapax Legomena Ratio | Programmatic | nltk | NLP |
| 3 | Average Sentence Length | Programmatic | nltk | NLP |
| 4 | Sentence Length Variance | Programmatic | nltk | NLP |
| 5 | Function Word Ratio | Programmatic | nltk | NLP |
| 6 | Long Word Ratio | Programmatic | stdlib | NLP |
| 7 | Flesch Reading Ease | Programmatic | textstat | NLP |
| 8 | Flesch-Kincaid Grade | Programmatic | textstat | NLP |
| 9 | Avg Syllables Per Word | Programmatic | textstat | NLP |
| 10 | Proselint Issue Count | Programmatic | proselint | NLP |
| 11 | POS Formality Score | Programmatic | spacy | NLP |
| 12 | Disfluency Ratio | Programmatic | spacy | NLP |
| 13 | Avg Dependency Depth | Programmatic | spacy | NLP |
| 14 | Noun / Verb Ratio | Programmatic | spacy | NLP |
| 15 | Avg Zipf Frequency | Programmatic | wordfreq | NLP |
| 16 | Rare Word Ratio | Programmatic | wordfreq | NLP |
| 17 | GPT-2 Perplexity | Programmatic | transformers | NLP |
| 18 | TF-IDF Template Score | Programmatic | sklearn | NLP |
| 19 | Register Gap | Programmatic | combined | NLP |
| 20 | Isolation Forest Score | Programmatic | sklearn | NLP |
| 21 | Z-Scores (per feature) | Programmatic | numpy | NLP |
| 22 | Composite Score | Programmatic | combined | NLP |
| 23 | Speaker Profile | LLM | OpenRouter | Semantic |
| 24 | Register Match | LLM | OpenRouter | Semantic |
| 25 | Vocabulary Match | LLM | OpenRouter | Semantic |
| 26 | Likely Origin | LLM | OpenRouter | Semantic |
| 27 | Specificity | LLM | OpenRouter | Semantic |
| 28 | LLM Markers Found | LLM | OpenRouter | Semantic |
| 29 | Structural Pattern | LLM | OpenRouter | Semantic |
| 30 | Confidence Contradiction | LLM | OpenRouter | Semantic |
| 31 | Cross-Answer Consistency | LLM | OpenRouter | Semantic |
| 32 | Pattern Description | LLM | OpenRouter | Semantic |
| 33 | Overall Assessment | LLM | OpenRouter | Semantic |
