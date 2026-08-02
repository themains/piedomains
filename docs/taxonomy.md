# Taxonomy design: why one list cannot work

This is a design document, not a description of what ships. `piedomains` currently emits
one label from a flat list of 44 (`piedomains.constants.classes`). This argues that the
flat list is the wrong shape, that the evidence for it is already in this repository, and
that every comparable taxonomy in the industry reached the same conclusion and acted on it.

**Outcome: the faceted redesign below was considered and declined.** The 44 classes stay
as one flat list, and the non-MECE-ness is handled by **multi-label output** instead --
every result row carries a `categories` list of every label above a probability threshold,
which is what IAB (arrays) and Cloudflare (multi-label, capped at two) both do. That needs
no retraining, no new label space, and no change to what `category` means, and it lifts the
chance of reporting the correct label from 79.7% to 86.6% at 1.35 labels per domain.

The argument below still stands and is why multi-label is the right answer rather than a
patch: the axes are real, they do collide, and a single slot cannot hold them. What is
declined is *restructuring the label space* to separate them.

The last section says what a faceted design would cost and what cannot yet be measured.

## 1. The label set has been repaired three times for the same reason

| release | what was removed or split | stated reason |
|---|---|---|
| v0.10 | `adv`, `tracker`, `spyware`, `redirector` | describe *who runs a site*, not what it says |
| v0.12 | domain parking given its own class | parking pages were 42% of `drugs` |
| this cycle | `games-misc`/`games-online`, `radiotv`/`webradio`, `downloads`/`warez` | ask about *delivery* or *legality*, not subject |

Each fix was correct and each was local. The v0.10 changelog had already named the general
case:

> The 39 classes conflated three unrelated questions — what a site is about, whether you
> would block it, and how it is built and monetised — into one mutually-exclusive softmax.
> `amazon.com` is a shopping site that also runs one of the largest ad networks on the web;
> forced to choose, the model picked `adv` and we scored it wrong.

The structure was diagnosed and the worst instance was treated. The structure remained.

## 2. A flat list that mixes questions cannot be MECE

Not "is hard to make MECE" — *cannot be*, as a matter of shape.

Take a shop selling gardening supplies. `gardening` and `shopping` are both true of
it. So the list fails **mutual exclusivity**: two labels apply. It also fails
**exhaustiveness**: no available label expresses what the site is, because the thing it is
requires two words from two different vocabularies. An annotator picks one and a second
annotator picks the other; whichever the model learns, it is marked wrong roughly half the
time on that shape of site.

Adding classes cannot repair this. A `garden_shopping` class fixes one cell and leaves
every other topic × commerce intersection open — and the same argument then applies to
topic × forum, topic × streaming, risk × commerce. The cross-product is the problem.

**MECE is only definable within a single question.** Ask one question, and a complete
non-overlapping set of answers exists. Ask three at once, and it does not.

### The measurement

`reports/taxonomy_v012.json` records 947 errors on 4,673 held-out documents disjoint from
training. Taking its top-25 conflated pairs — pairs of classes that trade errors in *both*
directions, which is the signature of a distinction the text does not carry — and asking
of each whether its two classes answer the same question:

| | pairs | errors | share of all errors |
|---|---|---|---|
| cross-axis (topic vs form, risk vs form) | 14 | 141 | 14.9% |
| delivery splits, fixed this cycle | 2 | 95 | 10.0% |
| legality split, fixed this cycle | 1 | 10 | 1.1% |
| **axis conflations, total** | **17** | **246** | **26.0%** |
| genuine same-question confusion | 8 | 76 | 8.0% |

**A quarter of all errors are the taxonomy asking two questions at once.** This is a lower
bound: it counts only the top 25 pairs, and the classification of each pair is a judgement
call (recorded in the script that produced the table, not asserted here).

Two classes dominate:

- **`shopping` appears in 11 of the 25 pairs** — with `automobile`, `travel`,
  `homestyle`, `gardening`, `pets`, `restaurants`,
  `sports`, `games`, `drugs`, `adult`, `news`. It is not a topic. It is
  what a site *does*, and it therefore collides with every topic a site can do it about.
- **`news` appears in 6.** Also not a topic — a news site is *about* politics or sport or
  finance. IAB classifies `News` under content *purpose*, not aboutness, for this reason.

The three splits fixed this cycle were the same error, which is why fixing them worked.

## 3. Everyone else already concluded this

### IAB Tech Lab — four taxonomies, and five orthogonal vectors inside one of them

IAB publishes separate taxonomies for separate questions: **Content** (what is this
about), **Audience** (who is the person), **Ad Product** (what does this creative sell),
and **Privacy**. Their implementation guide states the internal split directly:

> The 2.x & 3.x content taxonomy includes two parts – a set of categories that describe the
> topic context or **"aboutness"**, and an additional set of **orthogonal content
> attributes / "vectors"** such as content language, format, language, source, media type,
> etc.

The vectors are `Content Environment` (Email, Forum/Community, Marketplace/eCommerce,
Search Engine/Listings, Social, Utility/Online Tool, General), `Content Purpose`
(Informational→News/Educational/Review, Entertainment, Commerce, Conversational),
`Content Source`, `Content Form Factor` and `Brand Suitability and Risk`.

Note the proportions: **topic is ~705 nodes over four tiers; the five vectors together are
~47 rows, all flat.** Depth belongs to the topical axis and nowhere else.

**The decisive precedent** is what happened to their first attempt. Content Taxonomy 1.0
was one taxonomy used for content, ads and blocking simultaneously. IAB deprecated it in
2020 as *"not fit for purpose for any of its current use cases"* and split it into the
three that replaced it. That is this project's failure mode, at industry scale, already run
to its conclusion.

- <https://github.com/InteractiveAdvertisingBureau/Taxonomies/blob/main/implementation.md>
- <https://iabtechlab.com/standards/content-taxonomy/>

### GARM — harm and severity are two axes, not one

The Brand Safety Floor + Suitability Framework is a matrix: ~12 harm **categories** × four
**severity** levels (Floor / High / Medium / Low). Severity is not a property of the topic
but of the *treatment* — for Arms & Ammunition, "Promotion and advocacy of Sales of illegal
arms" is Floor while "Educational, Informative, Scientific treatment of Arms use" is Low.

IAB wired it in by putting the harm categories in the topic tree under `Sensitive Topics`
and the severity in a separate vector, explicitly:

> The **risk levels** … are treated as **additional attributes of the content. They are
> encoded in an orthogonal vector accordingly, allowing 'risk' to be associated with a
> 'topic' dynamically.**

GARM was discontinued by the WFA in August 2024; cite it as a design precedent rather than
a live standard.

- <https://wfanet.org/knowledge/item/2022/06/17/GARM-Brand-Safety-Floor--Suitability-Framework-3>

### Cloudflare — three groups, and the one it failed to factor out

Cloudflare splits by decision-semantics rather than subject: **Content categories**
(vendor-supplied topics), **Security risks** (model-derived from domain age and
reputation), **Security threats** (intelligence feeds). Domains carry multiple categories,
capped at two content ones.

The instructive part is the failure. "This domain has no real site" is scattered across all
three groups:

| condition | where Cloudflare files it |
|---|---|
| Parked & For Sale Domains | Security **risks** |
| No Content, Redirect, Unreachable, Login Screens | Content → **Miscellaneous** |
| DGA Domains | Security **threats** |

One concept, three homes, because status was never named as its own axis. Their
`Login Screens` definition even concedes the overlap — "sites hosting login screens **that
might also be included in other categories**".

- <https://developers.cloudflare.com/cloudflare-one/traffic-policies/domain-categories/>

### Curlie — four facets flattened into a tree, then patched

Curlie is topic × geography (`Regional`) × language (`World`) × audience-rating (`Adult`,
`Kids and Teens`). The `Adult` branch **clones the entire topic axis plus Regional plus
World underneath itself** — 14 subcategories mirroring the top level. `Kids and Teens` is
described in their own guidelines as "an entirely separate directory", with sites permitted
to appear in both.

Because a tree cannot hold four facets, the flattening is patched with `@link` symlinks,
"related category" backlinks, cross-language "language groups", and a short list of
sanctioned dual-listings. `Home/Gardening/Regional` is a category that holds **no sites at
all** — it exists purely to point into the geography axis.

- <https://curlie.org/docs/en/guidelines/site-specific.html>
- <https://curlie.org/docs/en/guidelines/regional/template.html>

### Two opposite answers worth knowing about

For a page genuinely about several things, Curlie **generalises upward** — list it at the
common ancestor, never in each child. Cloudflare **multi-labels**, capped at two. Neither
is obviously right; they are the two available moves once you admit the page has more than
one true label.

## 4. Proposed structure: one gate, three axes

### Axis 0 — status (gate; single-valued; decided first)

`live` · `parked` · `unavailable`

If status is not `live`, the other axes are **undefined — not unknown**. There is no site,
so there is nothing for it to be about. This is precisely the factoring Cloudflare never
performed.

The boundary against `outcomes.py` must stay sharp, and it is **not** "is there a site" but
**"did we get a page to read"**:

| | example | modelled as |
|---|---|---|
| no page retrieved | DNS failure, timeout, bot wall, under the token floor | **outcome** (`piedomains.outcomes`) |
| page retrieved, and it says there is no site | "this domain is for sale", `Index of /`, "account suspended" | **status** |

`blocking.looks_parked` and `blocking.looks_unavailable` already implement exactly this
axis, and `parked` scores F1 0.992 — the best class in the model — because a question with
one clean answer is learnable.

### Axis 1 — topic (single-valued; the only deep axis)

What the site is *about*. Around 20 values, modelled on IAB Tier 1 and pruned to what this
corpus can support:

`arts_and_entertainment` · `automotive` · `business_and_finance` · `careers` · `education` ·
`food_and_drink` · `games` · `health` · `home_and_garden` · `law_and_government` ·
`military` · `pets` · `politics` · `real_estate` · `religion_and_spirituality` · `science` ·
`society_and_culture` · `sports` · `technology` · `travel`

This is the axis that earns hierarchy later. The others should stay flat.

### Axis 2 — form (single-valued; shallow)

What kind of site it is, independent of subject. IAB's `Content Environment` plus the
`News` and `Commerce` values of `Content Purpose`:

`editorial` · `news` · `commerce` · `forum` · `social_network` · `search_directory` ·
`streaming` · `file_hosting` · `tool_or_utility` · `reference`

This is where `shopping`, `news`, `forum`, `socialnet`, `searchengines`, `radiotv`,
`downloads`, `imagehosting`, `webmail`, `urlshortener` and `library` actually belong.
Moving them off the topic axis is what dissolves the 11-of-25 and 6-of-25 above.

### Axis 3 — risk (shallow; "would you block it")

`none` · `adult` · `gambling` · `drugs` · `alcohol_tobacco` · `weapons` · `violence_or_hate`

GARM-style severity (Floor/High/Medium/Low) is a natural later extension and is
deliberately **not** proposed now: it multiplies annotation cost and nothing in the current
data supports it.

**Carried forward without change from v0.10: no identity categories.** Cloudflare files
LGBTQ beside `Lingerie & Bikini` and `Swimsuits`; filtering systems that treat identity as
sexual content have a documented history of harm. Splitting the taxonomy into axes must not
become an occasion to reintroduce one.

## 5. The principles

1. **One question per axis.** Two labels that answer different questions cannot share a
   softmax. This is the whole document in one line.
2. **MECE within an axis; multi-label across axes.** "MECE" applied to a mixed list is a
   category error.
3. **Is it visible in the page text?** — kept from `training/taxonomy.py`, and still the
   rule that excluded `adv`/`tracker`/`spyware`. Now paired with: **which question does it
   answer?**
4. **Status gates the rest.** No site ⇒ the other axes are undefined, not unknown.
5. **Depth belongs to topic.** IAB's ratio — 705 topical nodes, 47 vector rows — is the
   discipline.
6. **The "and" test assigns the axis.** If "it is X *and* Y" can be said of one page without
   contradiction, X and Y are on different axes. *A shop and about gardening* — different
   axes. *News and radio* — different axes, which is why that pair costs 20 errors.
7. **Do not encode what the page does not state.**
8. **No identity categories.**

## 6. The remap, and what the data can actually supervise

All 44 current classes, with the axes each would carry. `?` = the label is silent on that
axis; `–` = undefined because there is no site.

| current class | status | topic | form | risk |
|---|---|---|---|---|
| `adult` | live | ? | ? | adult |
| `alcohol` | live | food_and_drink | ? | alcohol_tobacco |
| `automobile` | live | automotive | ? | none |
| `dating` | live | society_and_culture | ? | none |
| `downloads` | live | ? | file_hosting | none |
| `drugs` | live | ? | ? | drugs |
| `education` | live | education | ? | none |
| `finance` | live | business_and_finance | ? | none |
| `fortunetelling` | live | religion_and_spirituality | ? | none |
| `forum` | live | ? | forum | none |
| `gamble` | live | ? | ? | gambling |
| `government` | live | law_and_government | ? | none |
| `cooking` | live | food_and_drink | ? | none |
| `games` | live | games | ? | none |
| `gardening` | live | home_and_garden | ? | none |
| `pets` | live | pets | ? | none |
| `homestyle` | live | home_and_garden | ? | none |
| `hospitals` | live | health | ? | none |
| `imagehosting` | live | ? | file_hosting | none |
| `isp` | live | technology | ? | none |
| `jobsearch` | live | careers | ? | none |
| `library` | live | ? | reference | none |
| `military` | live | military | ? | none |
| `movies` | live | arts_and_entertainment | ? | none |
| `music` | live | arts_and_entertainment | ? | none |
| `news` | live | ? | news | none |
| `politics` | live | politics | ? | none |
| `radiotv` | live | arts_and_entertainment | streaming | none |
| `realestate` | live | real_estate | ? | none |
| `humor` | live | society_and_culture | ? | none |
| `restaurants` | live | food_and_drink | ? | none |
| `sports` | live | sports | ? | none |
| `travel` | live | travel | ? | none |
| `wellness` | live | health | ? | none |
| `religion` | live | religion_and_spirituality | ? | none |
| `science` | live | science | ? | none |
| `searchengines` | live | ? | search_directory | none |
| `shopping` | live | ? | commerce | none |
| `socialnet` | live | ? | social_network | none |
| `urlshortener` | live | ? | tool_or_utility | none |
| `weapons` | live | ? | ? | weapons |
| `webmail` | live | ? | tool_or_utility | none |
| `parked` | parked | – | – | – |
| `unavailable` | unavailable | – | – | – |

### The column of question marks is the finding

Count what each axis would be trained on:

| axis | supervised by | masked | undefined |
|---|---|---|---|
| status | 44 of 44 | 0 | 0 |
| topic | 28 | 14 | 2 |
| **form** | **11** | **31** | 2 |
| risk | 42 (but only **5** carry a non-`none` value) | 0 | 2 |

**Shallalist supervises one axis per document.** `shopping` fixes form and says nothing
about topic; `automobile` fixes topic and says nothing about form; `porn` fixes risk and
says nothing about either.

This is not an obstacle to the design — **it is the same diagnosis arriving from the data
side.** A label set whose members answer different questions is a mixed-axis taxonomy, and
the question marks are where the mixing shows.

The intended handling is **masked loss**: each label supervises the axis it names, the
others are excluded from the loss for that document. Nothing is invented. The costs must be
stated plainly:

- **No axis gets full supervision.** `form` is the extreme case — 11 of 44 classes say
  anything about it, so a form head would be trained on roughly a quarter of the corpus.
- **Per-axis accuracy will not be comparable to today's 0.797.** Different question,
  different denominator. Anyone comparing the two numbers directly will be wrong.

## 7. What this document does not establish

**There is no gold multi-axis evaluation set, so this design cannot currently be validated
the way the merges were.** The merges were defensible because relabelling the shipped
model's own predictions gave a floor of +0.023/+0.023 before anything was retrained. No
equivalent exists here: Curlie is single-label, `tests/eval/labels.csv` is single-label and
covers only 17 of 44 classes, and Shallalist is single-label by construction.

Building one is the prerequisite for any claim of improvement — a few hundred domains
labelled on all three axes, by hand or by an LLM pass with human adjudication of
disagreements. Until that exists, this document is an argument from structure and from a
26% error-share measurement, and should be read as exactly that.

**Doing nothing is a legitimate option.** The flat taxonomy is wrong in a way that costs a
measurable quarter of errors, and it also works well enough to ship: `parked` at F1 0.992,
`religion` 0.946, `science` 0.958. The axes could be adopted one at a time — status is
already built and needs only to be *named* as an axis, and `form` is where the measured
error is concentrated. Nothing here requires the whole design to be taken at once.
