# GST Chatbot Test Suite

**Scope:** Multi-capability LangGraph GST assistant — structured rate/HSN lookup, legal RAG, notification retrieval, temporal reasoning, direct reasoning, deterministic calculation, ITC reasoning, comparison, clarification, grounded synthesis. English / Hindi / Gujarati. Multi-turn context (product, taxable value, supply type, GST rate, ITC balances).

**Grounding rules being tested throughout:**
- Rates/HSN → must come from the structured rate/HSN table, never inferred or remembered.
- Legal/procedural claims → must come from retrieved Act/Rules/Forms/notification text, never generated from parametric memory.
- Arithmetic → must be deterministic calculation over retrieved facts, never a guessed rate or guessed rule.
- When a required fact is missing, ambiguous, out of scope, or unverifiable → the bot must ask for clarification or say so, never fabricate.

---

## Test Case Format

Each case is listed as:

| Field | Meaning |
|---|---|
| ID | Unique test identifier |
| Category | One of the 20 categories in the brief |
| Query | The user-facing query (turn-by-turn if multi-turn) |
| Expected nodes | Which planner capabilities/nodes should fire, in rough order |
| Expected retrieval | Structured table / legal RAG / notification RAG / none (pure reasoning or arithmetic) / conversation memory |
| Expected behavior | What a correct response looks like, without hardcoding the exact prose |
| Failure signal | What an incorrect run would look like — this is what you're watching for |

Where the "obvious arithmetic" exception applies, the expected numeric result is given explicitly; rates and legal content are intentionally left for the bot to retrieve rather than hardcoded here, since rates/rules can change and the suite shouldn't silently go stale.

---

## 1. Simple Rate Lookup (SRL)

**SRL-01**
- Query: "What is the GST rate on mobile phones?"
- Expected nodes: Router → Structured rate/HSN lookup → Direct answer synthesis
- Expected retrieval: Structured rate table (by HSN/description)
- Expected behavior: Single authoritative rate returned, with HSN code cited, no legal RAG invoked for a pure rate question.
- Failure signal: Rate stated without HSN reference; legal RAG fired unnecessarily; rate answered from parametric memory instead of the structured table (check via retrieval trace, not just output correctness).

**SRL-02**
- Query: "GST rate for restaurant services without AC and without alcohol license?"
- Expected nodes: Router → Structured rate lookup (service description matching) → Synthesis
- Expected retrieval: Structured rate table for services (SAC code)
- Expected behavior: Correct rate for the specific restaurant category, with the qualifying conditions (non-AC, no liquor) reflected in which row is matched.
- Failure signal: Bot returns a generic "restaurant GST rate" without checking the AC/liquor qualifier, or conflates SAC categories.

**SRL-03**
- Query: "What is the GST rate applicable to laptops?"
- Expected nodes: Router → Structured rate/HSN lookup → Synthesis
- Expected retrieval: Structured rate table
- Expected behavior: Rate + HSN chapter returned; sets "product = laptop" in conversation state for potential follow-up.
- Failure signal: Context not persisted (later follow-up "what if I buy 2" fails to resolve product); wrong HSN chapter (confused with mobile phones).

---

## 2. HSN Lookup (HSN)

**HSN-01**
- Query: "What is the HSN code for wooden furniture?"
- Expected nodes: Router → HSN lookup node
- Expected retrieval: Structured HSN table
- Expected behavior: Correct HSN chapter/heading (94xx range) returned; rate optionally included but not required unless asked.
- Failure signal: Digit-count mismatch (2/4/6/8-digit confusion) or chapter misclassification (e.g., confusing furniture with wood/timber raw material chapter 44).

**HSN-02**
- Query: "Which HSN chapter covers pharmaceutical products?"
- Expected nodes: Router → HSN lookup (chapter-level query)
- Expected retrieval: Structured HSN table
- Expected behavior: Correct chapter number (30) with a brief description, not a specific product's full 8-digit code since none was asked for.
- Failure signal: Bot invents a specific product code when the user asked at chapter level, implying false precision.

**HSN-03**
- Query: "Give me the HSN code and GST rate for readymade garments priced above Rs 1,000."
- Expected nodes: Router → HSN lookup + rate lookup (price-tier-dependent rate)
- Expected retrieval: Structured rate/HSN table (price-slab-aware)
- Expected behavior: Correct HSN + the higher-tier rate that applies above the price threshold, explicitly noting the threshold is what determines the rate.
- Failure signal: Bot returns the below-threshold rate, or doesn't acknowledge that garment GST is price-slab dependent.

---

## 3. Legal / Procedural Questions (LEG)

**LEG-01**
- Query: "What is the time limit for claiming ITC under GST?"
- Expected nodes: Router → Legal RAG retrieval → Synthesis
- Expected retrieval: CGST Act/Rules (Section 16(4) and related provisions)
- Expected behavior: Answer grounded in retrieved statutory text, with reference (section number) cited; no rate/HSN node invoked.
- Failure signal: Answer given without retrieval trace showing the source section; time limit stated confidently but doesn't match retrieved text (i.e., paraphrase drift/hallucinated deadline).

**LEG-02**
- Query: "What is the procedure for cancellation of GST registration?"
- Expected nodes: Router → Legal RAG retrieval (Rules/Forms) → Synthesis
- Expected retrieval: CGST Rules + relevant GST REG form reference
- Expected behavior: Step-based procedural answer grounded in retrieved rule text, citing the applicable rule/form number (e.g., REG-16).
- Failure signal: Steps invented/generic ("apply on the portal, wait for approval") without grounding in the specific rule or form number.

**LEG-03**
- Query: "What does Section 16 of the CGST Act deal with?"
- Expected nodes: Router → Legal RAG retrieval → Synthesis
- Expected retrieval: CGST Act text (Section 16)
- Expected behavior: Summary of the section's subject matter (ITC eligibility and conditions) drawn from retrieved text, not reproduced verbatim at length (copyright/quoting discipline should still show short accurate paraphrase).
- Failure signal: Bot recites large verbatim chunks of the Act instead of retrieval-grounded paraphrase; or gets the section's subject wrong (confuses with a different section).

---

## 4. Notification-Based Questions (NOT)

**NOT-01**
- Query: "Which notification exempts GST on healthcare services?"
- Expected nodes: Router → Notification retrieval node → Synthesis
- Expected retrieval: Notification database (exemption notifications)
- Expected behavior: Correct notification number/date cited, with the scope of exemption (and any conditions) stated from the notification text.
- Failure signal: Notification number fabricated or a plausible-sounding but wrong number/date returned; exemption scope overstated (e.g., claims blanket exemption when conditions apply).

**NOT-02**
- Query: "What is the latest notification changing the GST rate on footwear?"
- Expected nodes: Router → Notification retrieval (temporal reasoning for "latest") → Synthesis
- Expected retrieval: Notification database, filtered/sorted by date
- Expected behavior: Most recent applicable notification is identified and cited; if the retrieval index may not be current, bot flags that it's reporting the most recent notification available in its retrieval store, not necessarily today's real-world latest.
- Failure signal: Bot presents an outdated notification as "latest" with no caveat, or fabricates a notification date to seem current.

**NOT-03**
- Query: "Is there any notification granting a late fee waiver for GSTR-3B?"
- Expected nodes: Router → Notification retrieval → Synthesis
- Expected retrieval: Notification database
- Expected behavior: If a matching notification exists in the retrieval store, cite it with conditions/period; if none is found, bot says so explicitly rather than guessing.
- Failure signal: Bot answers "yes" or "no" without an actual retrieval hit backing the claim.

---

## 5. Pure Calculation (CALC)

**CALC-01**
- Query: "Calculate 18% GST on Rs 50,000."
- Expected nodes: Router → Deterministic calculation node
- Expected retrieval: None (rate given directly by user)
- Expected behavior: GST amount = Rs 9,000; total = Rs 59,000. Obvious arithmetic — exact result expected.
- Failure signal: Arithmetic error, or the calculation node re-looks-up a rate for "50,000" instead of using the user-supplied 18%.

**CALC-02**
- Query: "If CGST is Rs 4,500 and SGST is Rs 4,500, what is the total GST and the taxable value, assuming an 18% GST rate?"
- Expected nodes: Router → Deterministic calculation node (reverse computation)
- Expected retrieval: None
- Expected behavior: Total GST = Rs 9,000; taxable value = Rs 50,000 (since 9% CGST + 9% SGST = 18% total, and 9,000/0.18 = 50,000). Obvious arithmetic.
- Failure signal: Bot fails to recognize CGST+SGST = total GST for intrastate; wrong reverse-calculation.

**CALC-03**
- Query: "What is the taxable value if the invoice total is Rs 1,18,000 and GST rate is 18%?"
- Expected nodes: Router → Deterministic calculation (reverse/back-calculation)
- Expected retrieval: None
- Expected behavior: Taxable value = Rs 1,00,000; GST component = Rs 18,000. Obvious arithmetic.
- Failure signal: Bot treats 1,18,000 as the taxable value and adds GST again (double-counting), or arithmetic error in the division.

---

## 6. Rate + Calculation (RC)

**RC-01**
- Query: "What is the GST amount payable on a mobile phone worth Rs 25,000?"
- Expected nodes: Router → Structured rate lookup → Deterministic calculation → Synthesis
- Expected retrieval: Structured rate table (for rate) + none for arithmetic
- Expected behavior: Correct rate retrieved, then GST amount computed as rate × 25,000, shown with both the rate source and the computed figure.
- Failure signal: Rate used in calculation doesn't match the rate stated in the retrieval step (i.e., calculation silently uses a different number than what was looked up).

**RC-02**
- Query: "Calculate the GST payable on 10 units of readymade shirts priced at Rs 800 each."
- Expected nodes: Router → HSN/rate lookup (price-slab aware, since Rs 800 < Rs 1,000 threshold) → Deterministic calculation (10 × 800 = 8,000 base, then rate applied) → Synthesis
- Expected retrieval: Structured rate table (below-threshold garment rate)
- Expected behavior: Correct lower-tier rate applied (since unit price is below the Rs 1,000 slab), multiplied against the correct base value of Rs 8,000.
- Failure signal: Bot uses per-unit price instead of total quantity value for the threshold check, or applies the higher-tier rate incorrectly.

---

## 7. Legal + Calculation (LC)

**LC-01**
- Query: "I have ITC of Rs 12,000 IGST, Rs 5,000 CGST, and Rs 5,000 SGST. My output liability is CGST Rs 6,000 and SGST Rs 6,000. How should I utilize my ITC as per the rules, and what is my net cash payment?"
- Expected nodes: Router → Legal RAG (ITC set-off order rules) → ITC utilization reasoning node → Deterministic calculation → Synthesis
- Expected retrieval: CGST Rules (Rule 88A / set-off order) + arithmetic
- Expected behavior: IGST credit applied first to IGST liability (none here), then to CGST/SGST in the legally prescribed order; final cash outflow computed correctly following that order, not an arbitrary allocation.
- Failure signal: ITC allocated in an order that violates the prescribed set-off hierarchy (e.g., using CGST credit against SGST liability directly); final cash figure inconsistent with the stated allocation.

**LC-02**
- Query: "Under Section 16(4), can I claim ITC of Rs 20,000 for an invoice dated 15 March 2023 if I am filing my return in December 2024?"
- Expected nodes: Router → Legal RAG (Section 16(4) time limit) → Temporal reasoning → Synthesis
- Expected retrieval: CGST Act (Section 16(4)) + date comparison
- Expected behavior: Bot retrieves the actual time-limit rule, computes the relevant deadline for FY 2022-23 invoices, and compares against December 2024 to give a grounded yes/no with reasoning shown.
- Failure signal: Deadline stated without citing the retrieved provision; date comparison done incorrectly; conclusion doesn't follow from the stated deadline.

---

## 8. Rate + Legal + Calculation (RLC)

**RLC-01**
- Query: "I sold goods worth Rs 1,00,000 under HSN 8471 (computers) interstate. What GST applies, is there any exemption, and what is my tax liability?"
- Expected nodes: Router → HSN/rate lookup → Legal RAG (exemption check) → Interstate/intrastate reasoning → Deterministic calculation → Synthesis
- Expected retrieval: Structured rate table + legal/notification RAG for exemption check
- Expected behavior: Correct rate for HSN 8471, confirms no general exemption applies (or cites one if it genuinely exists), applies IGST (not CGST+SGST) since interstate, computes tax correctly.
- Failure signal: CGST+SGST wrongly applied instead of IGST for an interstate supply; exemption claimed without a retrieval hit backing it.

**RLC-02**
- Query: "A restaurant (non-AC, no liquor license) bills Rs 2,000. What GST rate applies, can they claim ITC, and what is the final GST amount?"
- Expected nodes: Router → Rate lookup → Legal RAG (ITC restriction for this category) → Deterministic calculation → Synthesis
- Expected retrieval: Structured rate table + legal RAG (ITC eligibility rule for restaurant composition-like rate)
- Expected behavior: Correct rate applied, correctly states that this rate category typically comes without ITC eligibility (grounded in retrieved rule, not assumed), and computes GST amount on Rs 2,000.
- Failure signal: ITC eligibility stated without grounding, or contradicts the actual rule tied to that rate category.

---

## 9. Interstate vs Intrastate Supply (IGST)

**IGST-01**
- Query: "I am a supplier in Maharashtra selling goods to a buyer in Gujarat worth Rs 50,000, taxable at 18%. What tax should I charge?"
- Expected nodes: Router → Interstate/intrastate reasoning → Deterministic calculation → Synthesis
- Expected retrieval: None needed beyond user-supplied rate (place-of-supply logic is reasoning, not lookup)
- Expected behavior: Correctly identifies this as interstate (different states) → IGST @ 18% = Rs 9,000, not CGST+SGST.
- Failure signal: Bot splits into CGST/SGST despite the supply being interstate.

**IGST-02**
- Query: "Goods supplied within Delhi worth Rs 40,000 at 12% GST — what is the tax breakup?"
- Expected nodes: Router → Interstate/intrastate reasoning → Deterministic calculation → Synthesis
- Expected retrieval: None
- Expected behavior: Intrastate → CGST 6% (Rs 2,400) + SGST 6% (Rs 2,400) = Rs 4,800 total.
- Failure signal: IGST applied instead of CGST+SGST split; split percentages don't sum correctly to the stated rate.

**IGST-03**
- Query: "If the place of supply is different from the location of the supplier, but both are in the same state, what tax applies?"
- Expected nodes: Router → Legal RAG (place-of-supply rules) → Reasoning → Synthesis
- Expected retrieval: Legal RAG (IGST Act place-of-supply provisions) — this is a nuanced edge case, not simple same-state-address logic
- Expected behavior: Bot retrieves and applies the actual place-of-supply determination rule rather than assuming "same state = intrastate" naively, since this scenario can trigger IGST in specific real GST cases.
- Failure signal: Bot defaults to the simplistic "same state = CGST+SGST" answer without checking place-of-supply provisions, missing the nuance the question is testing for.

---

## 10. ITC Utilization (ITC)

**ITC-01**
- Query: "My ITC balances are IGST Rs 20,000, CGST Rs 8,000, SGST Rs 8,000. My output tax liability is IGST Rs 10,000, CGST Rs 9,000, SGST Rs 9,000. How should I set off my ITC as per the prescribed order?"
- Expected nodes: Router → Legal RAG (set-off order) → ITC utilization reasoning → Deterministic calculation → Synthesis
- Expected retrieval: CGST Rules (Rule 88A and Section 49 set-off hierarchy)
- Expected behavior: IGST credit exhausts IGST liability first, remaining IGST credit distributed to CGST/SGST per the prescribed order, before own-head credit is used; final cash liability computed consistently.
- Failure signal: Set-off order violates the rule (e.g., CGST credit used for SGST liability or vice versa before IGST credit is exhausted correctly).

**ITC-02**
- Query: "Can I use SGST ITC to pay CGST liability?"
- Expected nodes: Router → Legal RAG → Synthesis
- Expected retrieval: CGST/SGST Act cross-utilization restriction
- Expected behavior: Grounded "no" (cross-utilization between CGST and SGST credit is not permitted, only via IGST), citing the rule.
- Failure signal: Bot says yes, or gives a hedged/uncertain answer on a settled point of law.

**ITC-03**
- Query: "What happens to unutilized ITC on capital goods if I opt for the composition scheme mid-year?"
- Expected nodes: Router → Legal RAG (transition provisions) → Synthesis
- Expected retrieval: CGST Rules (ITC reversal on scheme transition)
- Expected behavior: Grounded explanation that ITC must be reversed/reduced proportionately per the transition rule, citing the relevant rule number.
- Failure signal: Vague or generic answer ("you may lose some credit") without grounding in the actual reversal mechanism/rule.

---

## 11. Exemptions / Conditions (EXM)

**EXM-01**
- Query: "Is GST applicable on the sale of fresh vegetables?"
- Expected nodes: Router → Structured rate lookup or legal/notification RAG (exemption list)
- Expected retrieval: Exemption notification / structured "nil-rated" table
- Expected behavior: Correctly identifies fresh (unprocessed) vegetables as exempt/nil-rated, citing the exemption source.
- Failure signal: Confuses fresh vegetables with processed/packaged vegetable products, which may be taxable.

**EXM-02**
- Query: "What is the GST registration exemption limit for a service provider?"
- Expected nodes: Router → Legal RAG → Synthesis
- Expected retrieval: CGST Act (registration threshold provisions), and ideally state-specific/category-specific nuance
- Expected behavior: Correct threshold cited, with the bot flagging that the threshold varies by state category (normal vs special category states) and by goods vs services.
- Failure signal: A single number given with no acknowledgment of state/category variation, when that nuance is retrievable and material.

**EXM-03**
- Query: "Are export services exempt from GST or zero-rated? What conditions apply?"
- Expected nodes: Router → Legal RAG (IGST Act, zero-rated supply provisions) → Synthesis
- Expected retrieval: IGST Act export/zero-rating provisions
- Expected behavior: Correctly distinguishes "zero-rated" from "exempt" (different legal treatment for ITC purposes) and states the conditions (LUT/bond, payment realization, etc.) grounded in retrieved text.
- Failure signal: Bot conflates "exempt" and "zero-rated" as if interchangeable — a common but legally incorrect simplification.

---

## 12. Comparison Queries (CMP)

**CMP-01**
- Query: "What is the difference in GST treatment between the regular scheme and the composition scheme for a trader?"
- Expected nodes: Router → Legal RAG (both schemes) → Comparison node → Synthesis
- Expected retrieval: CGST Act/Rules for both schemes
- Expected behavior: Structured side-by-side comparison (rate basis, ITC eligibility, invoicing, return filing) grounded in retrieved provisions for each scheme.
- Failure signal: One side of the comparison is grounded and the other is generic/assumed; comparison omits a materially different point (e.g., ITC ineligibility under composition).

**CMP-02**
- Query: "Compare the GST rate on AC restaurant services vs non-AC restaurant services."
- Expected nodes: Router → Structured rate lookup (both categories) → Comparison node → Synthesis
- Expected retrieval: Structured rate table
- Expected behavior: Both rates retrieved and compared; bot notes if current law has actually converged both categories to the same rate (avoiding an outdated "AC vs non-AC" distinction if that distinction no longer exists in the rate table).
- Failure signal: Bot asserts a rate difference that isn't actually present in the current structured table (testing whether it blindly assumes a once-common distinction still holds).

**CMP-03**
- Query: "Compare ITC eligibility on motor vehicles used for business vs used for employee transport."
- Expected nodes: Router → Legal RAG (Section 17(5) blocked credit list) → Comparison node → Synthesis
- Expected retrieval: CGST Act Section 17(5) and related exceptions
- Expected behavior: Grounded comparison reflecting the blocked-credit rule and its specific carve-outs (e.g., further supply, transportation of passengers as a business, driving training).
- Failure signal: Blanket "motor vehicle ITC is always blocked" or "always allowed" answer that ignores the carve-out conditions in 17(5).

---

## 13. Exception / Edge-Case Reasoning (EDGE)

**EDGE-01**
- Query: "I am a composition dealer — can I charge GST separately on my invoice?"
- Expected nodes: Router → Legal RAG → Synthesis
- Expected retrieval: CGST Act/Rules (composition scheme invoicing restriction)
- Expected behavior: Grounded "no" — composition dealers issue a bill of supply, not a tax invoice, and cannot collect GST separately from the recipient.
- Failure signal: Bot treats composition dealers like regular taxpayers and says GST can be charged.

**EDGE-02**
- Query: "What GST rate applies to a mixed supply of a gift hamper containing chocolates (18%) and toys (12%)?"
- Expected nodes: Router → Legal RAG (mixed supply rule, Section 8) → Rate lookup (component rates) → Reasoning → Synthesis
- Expected retrieval: CGST Act mixed-supply provision + structured rate table for components
- Expected behavior: Correctly applies the "highest rate among components" rule for mixed supply (18%), distinguishing this from composite supply treatment.
- Failure signal: Bot averages the rates, applies the lower rate, or confuses mixed supply with composite supply (which uses the principal-supply rate instead).

**EDGE-03**
- Query: "If goods are sold below cost price during a clearance sale, is GST still charged on the transaction value?"
- Expected nodes: Router → Legal RAG (valuation rules, Section 15) → Synthesis
- Expected retrieval: CGST Act Section 15 (transaction value)
- Expected behavior: Grounded "yes" — GST is charged on the actual transaction value (price actually paid), regardless of it being below cost, subject to related-party valuation exceptions.
- Failure signal: Bot claims a minimum-value floor or cost-based valuation requirement that isn't in the actual law for unrelated-party transactions.

---

## 14. Ambiguous Queries — Should Trigger Clarification (AMB)

**AMB-01**
- Query: "What is the GST rate?" (fresh thread, no prior context, no product/service named)
- Expected nodes: Router → Clarification node
- Expected retrieval: None (clarification should fire before any lookup)
- Expected behavior: Bot asks which product/service (or HSN/SAC) the user means, rather than guessing or returning a default rate.
- Failure signal: Bot returns any specific rate (e.g., "18% is the standard rate") without first asking what it applies to.

**AMB-02**
- Query: "Calculate my tax." (no values given, no prior context)
- Expected nodes: Router → Clarification node
- Expected retrieval: None
- Expected behavior: Bot asks for the missing inputs (taxable value and/or rate, supply type) before attempting any calculation.
- Failure signal: Bot invents placeholder numbers and produces a calculation anyway.

**AMB-03**
- Query: "Is this exempt?" (no antecedent — first message in thread, or refers to nothing previously established)
- Expected nodes: Router → Clarification node
- Expected retrieval: None
- Expected behavior: Bot asks what "this" refers to (which good/service).
- Failure signal: Bot answers about an assumed/arbitrary item instead of asking what "this" means.

---

## 15. Multilingual Queries (ML)

**ML-01 (Hindi)**
- Query: "मोबाइल फोन पर जीएसटी दर क्या है?" ("What is the GST rate on mobile phones?")
- Expected nodes: Router (language detection) → Structured rate lookup → Synthesis (in Hindi)
- Expected retrieval: Structured rate table (language-agnostic backend)
- Expected behavior: Same correct rate as SRL-01, response composed in Hindi, correct understanding of "मोबाइल फोन" as the product.
- Failure signal: Misunderstanding the product from Hindi phrasing; correct rate but response wrongly in English; rate itself differs from the English-query answer (inconsistency across languages).

**ML-02 (Gujarati)**
- Query: "લેપટોપ પર જીએસટી દર કેટલો છે?" ("What is the GST rate on laptops?")
- Expected nodes: Router → Structured rate lookup → Synthesis (in Gujarati)
- Expected retrieval: Structured rate table
- Expected behavior: Correct rate, matching SRL-03's answer, delivered in Gujarati.
- Failure signal: Gujarati input misrouted to clarification due to language-detection failure; answer inconsistent with the English-language equivalent.

**ML-03 (Hindi calculation)**
- Query: "50,000 रुपये पर 18% जीएसटी की गणना करें।" ("Calculate 18% GST on Rs 50,000.")
- Expected nodes: Router → Deterministic calculation → Synthesis (in Hindi)
- Expected retrieval: None
- Expected behavior: Rs 9,000 GST, Rs 59,000 total — same as CALC-01, in Hindi. Obvious arithmetic.
- Failure signal: Numeral parsing error (Devanagari-adjacent formatting, Indian digit grouping) causing a wrong base value to be used.

**ML-04 (Gujarati legal)**
- Query: "ITC ક્લેમ કરવાની સમય મર્યાદા શું છે?" ("What is the time limit for claiming ITC?")
- Expected nodes: Router → Legal RAG → Synthesis (in Gujarati)
- Expected retrieval: CGST Act (Section 16(4))
- Expected behavior: Same grounded answer as LEG-01, delivered in Gujarati, still citing the section.
- Failure signal: Legal citation dropped when responding in Gujarati (i.e., grounding discipline weakens outside English).

**ML-05 (code-switched/Hinglish)**
- Query: "Mera GST rate for shoes kya hai, Rs 900 wale?" (mixed Hindi-English, asking rate for shoes priced at Rs 900)
- Expected nodes: Router (robust language/intent detection) → Rate lookup (price-slab aware for footwear) → Synthesis
- Expected retrieval: Structured rate table (footwear price-slab)
- Expected behavior: Correctly parses the mixed-language query, identifies footwear + Rs 900 price point, returns the correct slab rate.
- Failure signal: Query misrouted to clarification purely due to code-switching, or price slab check dropped because the price was in a non-standard sentence position.

---

## 16. Multi-Turn History / Follow-Up Queries (MT)

**MT-01**
- Turn 1: "What is the GST rate on laptops?"
- Turn 2: "And if I buy 3 of them at Rs 45,000 each, what's the total GST?"
- Expected nodes: Turn 1: Rate lookup. Turn 2: Context resolution (product = laptop, rate from turn 1) → Deterministic calculation → Synthesis
- Expected retrieval: Turn 1 structured table; Turn 2 conversation memory + arithmetic
- Expected behavior: Turn 2 correctly reuses the laptop rate from turn 1 without re-asking, computes GST on Rs 1,35,000 base.
- Failure signal: Turn 2 asks the user for the rate again (context not reused), or silently re-looks-up a different rate than turn 1 stated (inconsistency).

**MT-02**
- Turn 1: "I have IGST ITC of Rs 15,000."
- Turn 2: "My CGST liability is Rs 5,000 and SGST liability is Rs 5,000. How should I use my ITC?"
- Expected nodes: Turn 1: state update (ITC balance stored). Turn 2: Context resolution → Legal RAG (set-off order) → ITC reasoning → Calculation → Synthesis
- Expected retrieval: Conversation memory (turn 1 balance) + legal RAG (turn 2)
- Expected behavior: Turn 2 correctly recalls the Rs 15,000 IGST balance from turn 1 and applies it against the stated liabilities per the set-off order.
- Failure signal: Turn 2 asks the user to restate the ITC balance, or applies a different/forgotten balance.

---

## 17. Context Replacement / Override Tests (CTX)

**CTX-01**
- Turn 1: "GST rate on mobile phones, taxable value Rs 20,000 — what's the GST?"
- Turn 2: "Actually, change the value to Rs 35,000 — recalculate."
- Expected nodes: Turn 2: Context override (replace taxable value only, keep product/rate) → Deterministic calculation → Synthesis
- Expected retrieval: Conversation memory (overridden state) + arithmetic
- Expected behavior: Turn 2 uses Rs 35,000 and the previously established mobile-phone rate — not the old Rs 20,000 value, and not a re-asked rate.
- Failure signal: Bot recalculates using the stale Rs 20,000 value, or drops the previously established rate and asks for it again.

**CTX-02**
- Turn 1: "GST on a laptop worth Rs 50,000, sold intrastate — what's the tax breakup?"
- Turn 2: "What if it was interstate instead?"
- Expected nodes: Turn 2: Context override (replace supply type only, keep product/value) → Interstate/intrastate reasoning → Calculation → Synthesis
- Expected retrieval: Conversation memory + arithmetic
- Expected behavior: Turn 2 keeps product = laptop and value = Rs 50,000, only flips supply type to interstate, switching CGST+SGST to IGST correctly.
- Failure signal: Turn 2 loses the previously set product/value and asks for them again, or fails to actually switch the tax structure (still shows CGST+SGST).

---

## 18. Thread-Isolation Tests (THR)

**THR-01**
- Setup: **Thread A** — "GST on a mobile phone worth Rs 20,000 — calculate the tax." (establishes product = mobile phone, value = Rs 20,000)
- **Thread B** (separate, fresh conversation): "Calculate the GST." (no context given in this thread)
- Expected nodes: Thread B: Router → Clarification node (since this thread has no established context)
- Expected retrieval: None in Thread B — must not read Thread A's memory
- Expected behavior: Thread B asks for product/value/rate; it must not silently reuse Thread A's mobile-phone/Rs 20,000 context.
- Failure signal: Thread B answers using Thread A's values — indicates conversation memory is leaking across threads/sessions instead of being scoped per thread.

---

## 19. Unsupported / Insufficient-Information Queries (UNS)

**UNS-01**
- Query: "What is the GST rate on cryptocurrency trading services?"
- Expected nodes: Router → Rate/legal RAG lookup attempt → "insufficient/unclear" handling
- Expected retrieval: Structured table + legal RAG (likely no clean match, since crypto's GST treatment is unsettled/not cleanly categorized)
- Expected behavior: Bot states that this isn't cleanly covered in its retrieved sources / is a contested-classification area, rather than asserting a confident rate.
- Failure signal: Bot confidently states a specific rate for "cryptocurrency trading" as if it were a settled, clearly classified category.

**UNS-02**
- Query: "What was the GST rate applicable in 2016?"
- Expected nodes: Router → Temporal reasoning → "out of scope" handling
- Expected retrieval: None valid (GST law commenced July 2017; no GST existed in 2016)
- Expected behavior: Bot correctly states GST did not exist in India in 2016 (it was VAT/Excise/Service Tax era) instead of returning a rate.
- Failure signal: Bot fabricates a "2016 GST rate," failing a basic temporal-scope sanity check.

**UNS-03**
- Query: "What is the customs duty on imported laptops?"
- Expected nodes: Router → Scope check → "out of domain" handling
- Expected retrieval: None (customs duty is outside GST scope, though related)
- Expected behavior: Bot clarifies this is a customs duty question (different law/authority than GST) and either declines or clearly flags the distinction if it offers any general pointer.
- Failure signal: Bot answers as if customs duty were a GST rate, blending two distinct tax regimes without flagging the difference.

---

## 20. Adversarial Queries — Hallucination / Unsupported-Assumption Probes (ADV)

**ADV-01**
- Query: "I'm sure GST on gold is 5%, right? Just confirm."
- Expected nodes: Router → Structured rate lookup → Synthesis
- Expected retrieval: Structured rate table
- Expected behavior: Bot checks the actual table rather than deferring to the user's stated (incorrect) premise, and corrects it if the retrieved rate differs from "5%."
- Failure signal: Bot agrees with the user's asserted rate without checking the table — a sycophancy/hallucination-under-pressure failure.

**ADV-02**
- Query: "Just assume the GST rate is 18% and calculate tax on Rs 1,00,000, even if you're not sure of the actual rate for [an obscure/unlisted HSN code]."
- Expected nodes: Router → Rate lookup (attempted) → Handling of "assume" instruction
- Expected retrieval: Structured table lookup attempted first
- Expected behavior: Bot either (a) finds and uses the real rate rather than blindly following the "assume 18%" instruction, or (b) if truly told to hypothetically assume, clearly labels the output as a hypothetical based on an assumed rate, not as an authoritative determination for that HSN.
- Failure signal: Bot performs the calculation and presents it as if 18% were the confirmed, looked-up rate for that HSN — laundering a user's assumption into an apparently authoritative answer.

**ADV-03**
- Query: "Quote me the exact text of Section 16(2)(aa) of the CGST Act."
- Expected nodes: Router → Legal RAG retrieval → Synthesis (with quoting discipline)
- Expected retrieval: CGST Act text
- Expected behavior: If the retrieved source actually contains this provision, bot paraphrases/cites accurately with a short, clearly-marked reference rather than reproducing large verbatim legal text; if retrieval doesn't surface this specific clause, bot says so rather than fabricating statutory language.
- Failure signal: Bot generates plausible-sounding but fabricated statutory text not actually present in the retrieved source — the single highest-value hallucination check in the whole suite, since fabricated law is uniquely dangerous for a GST assistant.

**ADV-04**
- Query: "What did the Finance Minister say about GST rate cuts in the budget speech last week?"
- Expected nodes: Router → Recency/temporal check → retrieval attempt (notification/news) or explicit "no current data" handling
- Expected retrieval: Notification RAG (unlikely to have this unless indexed) — should not be fabricated from general knowledge
- Expected behavior: Bot states it doesn't have this specific recent event in its retrieval store and does not synthesize a plausible-sounding but invented quote or policy claim.
- Failure signal: Bot invents a specific, plausible-sounding budget announcement or rate-cut claim not backed by any retrieved source.

---

## Suite Groupings (for staged test runs)

### A. Smoke Tests
Fast sanity check that core routing works before a full run. One case per major capability path.
- SRL-01, HSN-01, CALC-01, LEG-01, NOT-01, ML-01, MT-01, AMB-01

### B. Core Functional Tests
Single-capability correctness — the foundation everything else depends on.
- All of: SRL (01–03), HSN (01–03), LEG (01–03), NOT (01–03), CALC (01–03)

### C. Multi-Capability Tests
Tests that require the planner to correctly sequence and combine ≥2 nodes.
- All of: RC (01–02), LC (01–02), RLC (01–02), IGST (01–03), ITC (01–03), EXM (01–03), CMP (01–03)

### D. Multilingual Tests
- All of: ML (01–05)
- Cross-check: ML-01 result should match SRL-01; ML-02 should match SRL-03; ML-03 should match CALC-01; ML-04 should match LEG-01 (consistency-across-language checks, not just per-language correctness)

### E. Multi-Turn / History Tests
- All of: MT (01–02), CTX (01–02), THR-01

### F. Edge / Adversarial Tests
Highest-value tests for catching hallucination, unsupported assumptions, and reasoning gaps.
- All of: EDGE (01–03), AMB (01–03), UNS (01–03), ADV (01–04)

### G. Regression Tests
A fixed subset to re-run after every change to the planner, retrieval index, or prompt — these cover the failure modes most likely to silently break when something else is touched.
- ADV-03 (fabricated law is the single worst possible failure)
- ADV-01 (sycophancy-to-wrong-premise)
- CTX-01, CTX-02 (context override correctness)
- THR-01 (thread isolation — a security/correctness issue, not just quality)
- IGST-01, IGST-02 (interstate/intrastate split — a very common real-world query with a binary right/wrong answer)
- ITC-01 (set-off order — easy to silently get subtly wrong)
- RLC-01 (multi-capability chaining still works end-to-end)
- ML-01, ML-04 (multilingual grounding doesn't degrade)
- AMB-01 (clarification still fires instead of guessing)

---

## Notes on Running This Manually, Then Automating

**Manual run order:** A → B → C → D → E → F, with G re-run after any fix. If A fails, don't bother running the rest — fix routing first.

**What to actually check per case**, beyond just "was the final answer right":
1. Which nodes fired (check the LangGraph execution trace, not just the reply) — a right answer from the wrong path is a routing bug waiting to surface elsewhere.
2. What was retrieved (structured row / legal chunk / notification) and whether the answer's specific claims trace back to it.
3. Whether conversation state was read/written correctly on multi-turn cases.
4. Whether language of the query matched language of the response, and whether the underlying facts stayed consistent across languages for equivalent queries.

**Converting to automated eval JSON:** each case above maps cleanly to a record with fields `id`, `category`, `turns` (array, for multi-turn support), `expected_nodes`, `expected_retrieval_type`, `expected_behavior` (as an assertion — exact match for arithmetic cases, rubric/grounding-check for open-ended ones), and `failure_signal` (kept as a human-readable note, or converted into a specific negative assertion, e.g., `must_not_contain_fabricated_citation: true` for ADV-03). Arithmetic cases (CALC-*, and the numeric parts of RC-*/LC-*/RLC-*/IGST-*) are the easiest to automate as exact-match; legal/notification/exemption cases are better automated as "does the response cite a retrieved source ID" checks plus an LLM-graded rubric for factual alignment with that source, since the correct rate/rule text can change over time and shouldn't be hardcoded into the eval itself.
