# OAEI Conference (ra1) — error analysis of the hybrid matcher

Date: 2026-10-04  ·  Commit: `954a65894888`  ·  Matcher: `olaya-hybrid[olaya-embedding[intfloat/e5-small-v2,both,csls10],w=0.50,cascade,alpha=0.00]` at threshold 0.50 (the a-priori hybrid row of `oaei-conference-ra1.md`: lexical cascade first, CSLS-calibrated e5-small-v2 cosine on the leftovers).

Every reference correspondence the system did not produce (false negative) and every produced correspondence that
is not in ra1 (false positive) over the 21 pairs, with the labels, comments (italic, when the ontology has one),
parents (classes) or `domain -> range` (properties), the matcher's score with the lexical and embedding scores in
parentheses, the rank of the gold target among the source entity's candidates (row) and of the gold source among
the target's candidates (col), the best candidate on each side and the stage at which the gold pair was lost:

- **cascade**: one of the two entities was consumed by the lexical stage with another partner (the pair was never scored);
- **scoring**: the gold target is not the entity's top-1 candidate (the score itself is wrong);
- **mutual-best**: the gold is the source entity's top-1 but another source entity wins the target's column;
- **threshold**: the pair is mutual-best but scores below 0.50.

Categories are descriptive and computed from the two labels only (identical label; inflection / spelling; head noun
shared, modifier differs; compound / partial overlap; abbreviation; property naming = properties without a shared
token; synonym / paraphrase = classes without a shared token).

## Summary

TP 193, FN 112 (classes 83, properties 29), FP 80 (classes 38, properties 42); 14 of the FPs block a gold correspondence (1:1 competition), the rest pair two entities ra1 leaves unmatched.

### False negatives by stage

| stage | FNs |
|---|---|
| cascade | 15 |
| scoring | 39 |
| mutual-best | 9 |
| threshold | 49 |

### False negatives by category (with the stage breakdown)

| category | all | classes | properties | cascade | scoring | mutual-best | threshold |
|---|---|---|---|---|---|---|---|
| compound / partial overlap | 36 | 29 | 7 | 8 | 6 | 3 | 19 |
| synonym / paraphrase | 30 | 30 | 0 | 4 | 17 | 1 | 8 |
| head noun shared, modifier differs | 24 | 24 | 0 | 1 | 4 | 3 | 16 |
| property naming | 22 | 0 | 22 | 2 | 12 | 2 | 6 |

### False negatives by score band (the matcher's score of the gold pair)

| band | FNs |
|---|---|
| 0.00 (unscored) | 32 |
| 0.01-0.19 | 27 |
| 0.20-0.29 | 17 |
| 0.30-0.39 | 21 |
| 0.40-0.49 | 15 |
| >= 0.50 | 0 |

### False positives by category

| category | FPs |
|---|---|
| identical label | 34 |
| compound / partial overlap | 23 |
| head noun shared, modifier differs | 9 |
| inflection / spelling | 9 |
| property naming | 4 |
| synonym / paraphrase | 1 |

### Reading

- **Recall is lost mostly at the threshold, not at the ranking.**  49 of 112 FNs are already the
  mutual-best pair and only fail the 0.50 cut-off; 36 FNs score between 0.30 and 0.49.  Almost all of these are
  *compound / head-noun* pairs (`Regular author` = `Author`, `Conference Event` = `Event`, `Accepted contribution` =
  `Accepted Paper`, `Topic` = `Research Topic`) whose lexical score is 0.35-0.49 (IDF-weighted Jaccard penalises the
  unshared modifier) and whose *embedding* score is lower still, and the cascade scores a leftover pair by the
  embedding alone.  The obvious fixes (a `max(lex, emb)` second stage, a head-noun field or bonus) were measured
  and do *not* help: every pair they lift has a twin ra1 rejects (`Paper Abstract` / `Abstract`, `Reviewer` /
  `Possible Reviewer`, `Regular` / `Regular Paper`).  What does help is giving the encoder the entity's *path*
  from the root (`Conference > Document > Paper`; `Paper > has Author > Person` for a property): the parent chain
  separates `Conference Venue Place` from `Conference hall` and `start Date` from `Start of conference`, and
  the domain / range disambiguates property names (see the ablations below).
- **Scoring failures (39) are synonyms the small encoder cannot see** (`Paper` / `Contribution`,
  `Event` / `Activity`, `Attendee` / `Delegate`, `Conference proceedings` / `Publication`: calibrated cosine 0.00-0.10)
  and properties whose names share nothing (`assignedTo` / `isReviewedBy`, `hasBeenAssigned` / `isReviewing`).
  Larger encoders and ensembles are the only lever for the former; the latter need domain / range reasoning.
- **Cascade losses (15) are mostly ra1 idiosyncrasies**: `Conference_volume` = `Conference` (five pairs)
  is lost because `conference:Conference` matches the other ontology's `Conference` exactly, which ra1 counts as
  wrong; `Paper` = `Contribution` is lost to `Paper` = `Paper` (confOf has both), `hasAuthor` = `writtenBy` to
  `writtenBy` = `writtenBy`.  No label-based matcher can recover these without losing the exact matches elsewhere.
- **Mutual-best losses (9)** are 1:1 competitions (`Program Committee Chair` = `Chair PC` loses to
  `Conference Chair`), where a global assignment (Hungarian) could in principle help.
- **Precision**: 34 of 80 FPs are *identical labels* that ra1 does not
  contain (`Paper` / `Paper` in cmt-conference, `Document` / `Document`, `Conference` / `Conference`), and another
  9 are inflections (`has members` / `has Member`).  They are the
  precision ceiling of any string-based matcher on ra1 and are unrelated to the recall problem.
- Categories: compound / partial overlap 36, synonym / paraphrase 30, head noun shared, modifier differs 24, property naming 22.  Comments are too sparse to derive
  synonyms from (cmt 4, confOf 54 (all the string `confTool`), edas 10, ekaw 2, none elsewhere; no rdfs:labels at all),
  so the only honest sources of synonymy are the encoders.

## False negatives (all pairs)

| pair | kind | source | target | score (lex / emb) | rank row / col | row best | col best | lost at | category |
|---|---|---|---|---|---|---|---|---|---|
| cmt-conference | class | **Author** `Author` [Conference Member; User] | **Regular author** `Regular_author` [Conference contribution; Conference contributor; Contribution 1th author; Contribution co author] | 0.30 (0.39 / 0.30) | 1 / 1 | Regular author (0.30) | Author (0.30) | threshold | head noun shared, modifier differs |
| cmt-conference | class | **Chairman** `Chairman` [Person] | **Chair** `Chair` [Committee member] | 0.00 (0.00 / 0.46) | - / - | Co chair (0.25) | Conference Chair (0.52) | cascade (Conference Chair -> Chair) | synonym / paraphrase |
| cmt-conference | class | **Conference** `Conference` [-] | **Conference volume** `Conference_volume` [Conference] | 0.00 (0.42 / 0.04) | - / - | Conference (1.00) | - | cascade (Conference -> Conference) | compound / partial overlap |
| cmt-conference | class | **Subject Area** `SubjectArea` [-] | **Topic** `Topic` [-] | 0.31 (0.00 / 0.31) | 1 / 1 | Topic (0.31) | Subject Area (0.31) | threshold | synonym / paraphrase |
| cmt-conference | property | **assign External Reviewer** `assignExternalReviewer` [Reviewer -> External Reviewer] | **invites co reviewers** `invites_co-reviewers` [Reviewer -> Reviewer] | 0.04 (0.16 / 0.04) | 3 / 4 | belongs to reviewers (0.10) | start Reviewer Bidding (0.10) | scoring | compound / partial overlap |
| cmt-conference | property | **assigned By Reviewer** `assignedByReviewer` [External Reviewer -> Reviewer] | **invited by** `invited_by` [Reviewer -> Reviewer] | 0.00 (0.24 / 0.00) | - / - | has been assigned a review reference (0.32) | added By (0.21) | scoring | property naming |
| cmt-confOf | class | **Conference Member** `ConferenceMember` [Person] | **Member** `Member` [Participant] *confTool* | 0.42 (0.42 / 0.42) | 1 / 1 | Member (0.42) | Conference Member (0.42) | threshold | head noun shared, modifier differs |
| cmt-confOf | class | **Paper** `Paper` [Document] | **Contribution** `Contribution` [-] *confTool* | 0.00 (0.00 / 0.00) | - / - | Paper (0.84) | - | cascade (Paper -> Paper) | synonym / paraphrase |
| cmt-confOf | class | **Paper Full Version** `PaperFullVersion` [Paper] | **Paper** `Paper` [Contribution] *confTool* | 0.00 (0.21 / 0.30) | - / - | Short paper (0.25) | Paper (0.84) | cascade (Paper -> Paper) | compound / partial overlap |
| cmt-confOf | class | **Program Committee Chair** `ProgramCommitteeChair` [Chairman; Program Committee Member] | **Chair PC** `Chair_PC` [Person] *confTool* | 0.22 (0.22 / 0.22) | 1 / 3 | Chair PC (0.22) | Conference Chair (0.26) | mutual-best | compound / partial overlap |
| cmt-confOf | class | **Program Committee Member** `ProgramCommitteeMember` [Conference Member; Person] *The Program Committee consists of all members of the confere* | **Member PC** `Member_PC` [Person] *Reviewer* | 0.20 (0.21 / 0.20) | 1 / 2 | Member PC (0.20) | Conference Member (0.31) | mutual-best | compound / partial overlap |
| cmt-confOf | class | **Subject Area** `SubjectArea` [-] | **Topic** `Topic` [-] *confTool* | 0.20 (0.00 / 0.20) | 1 / 1 | Topic (0.20) | Subject Area (0.20) | threshold | synonym / paraphrase |
| cmt-confOf | property | **has Author** `hasAuthor` [Paper -> Author] | **written By** `writtenBy` [Contribution -> Author] *confTool* | 0.00 (0.03 / 0.06) | - / - | has Surname (0.05) | written By (0.85) | cascade (written By -> written By) | property naming |
| cmt-confOf | property | **has Been Assigned** `hasBeenAssigned` [Reviewer -> Paper] | **reviewes** `reviewes` [Member PC -> Contribution] | 0.00 (0.00 / 0.00) | - / - | employed By (0.05) | write Review (0.31) | scoring | property naming |
| cmt-confOf | property | **has Subject Area** `hasSubjectArea` [Paper -> Subject Area] | **deals With** `dealsWith` [Contribution -> Topic] *confTool* | 0.01 (0.00 / 0.01) | 2 / 3 | has Topic (0.25) | assigned To (0.07) | scoring | property naming |
| cmt-edas | property | **assigned To** `assignedTo` [Paper -> Reviewer] | **is Reviewed By** `isReviewedBy` [Paper -> Conference Chair, Reviewer, Session Chair, TPC Member] | 0.00 (0.03 / 0.00) | - / - | provided By (0.03) | read By Reviewer (0.26) | scoring | property naming |
| cmt-edas | property | **has Author** `hasAuthor` [Paper -> Author] | **is Written By** `isWrittenBy` [Paper -> Author] | 0.00 (0.07 / 0.22) | - / - | has Related Paper (0.21) | written By (0.60) | cascade (written By -> is Written By) | property naming |
| cmt-edas | property | **has Been Assigned** `hasBeenAssigned` [Reviewer -> Paper] | **is Reviewing** `isReviewing` [Conference Chair, Reviewer, Session Chair, TPC Member -> Paper] | 0.00 (0.03 / 0.00) | - / - | has Rating (0.14) | enter Review Criteria (0.18) | scoring | property naming |
| cmt-ekaw | class | **Paper Full Version** `PaperFullVersion` [Paper] | **Regular Paper** `Regular_Paper` [Paper] | 0.24 (0.17 / 0.24) | 1 / 1 | Regular Paper (0.24) | Paper Full Version (0.24) | threshold | compound / partial overlap |
| cmt-ekaw | property | **assigned To** `assignedTo` [Paper -> Reviewer] | **has Reviewer** `hasReviewer` [Paper -> Possible Reviewer] | 0.00 (0.07 / 0.00) | - / - | organised By (0.02) | read By Reviewer (0.32) | scoring | property naming |
| cmt-ekaw | property | **has Been Assigned** `hasBeenAssigned` [Reviewer -> Paper] | **reviewer Of Paper** `reviewerOfPaper` [Possible Reviewer -> Paper] | 0.00 (0.06 / 0.00) | - / - | has Review (0.14) | reviews Per Paper (0.25) | scoring | property naming |
| cmt-ekaw | property | **written By** `writtenBy` [Review -> Reviewer] | **review Written By** `reviewWrittenBy` [Review -> Possible Reviewer] | 0.00 (0.71 / 0.39) | - / - | written By (0.87) | write Review (0.55) | cascade (written By -> written By) | compound / partial overlap |
| cmt-sigkdd | class | **Conference Chair** `ConferenceChair` [Chairman; Conference Member] | **General Chair** `General_Chair` [Organizator] | 0.31 (0.29 / 0.31) | 2 / 2 | Conference hall (0.32) | Chairman (0.36) | scoring | head noun shared, modifier differs |
| conference-confOf | class | **Conference contribution** `Conference_contribution` [Conference document] | **Contribution** `Contribution` [-] *confTool* | 0.11 (0.45 / 0.11) | 1 / 5 | Contribution (0.11) | Written contribution (0.24) | mutual-best | head noun shared, modifier differs |
| conference-confOf | class | **Conference volume** `Conference_volume` [Conference] | **Conference** `Conference` [Working event] | 0.00 (0.38 / 0.26) | - / - | - | Conference (0.87) | cascade (Conference -> Conference) | compound / partial overlap |
| conference-confOf | class | **Regular author** `Regular_author` [Conference contribution; Conference contributor; Contribution 1th author; Contribution co author] | **Author** `Author` [Person] *confTool* | 0.20 (0.40 / 0.20) | 2 / 1 | Regular (0.35) | Regular author (0.20) | scoring | head noun shared, modifier differs |
| conference-confOf | property | **has a track workshop tutorial topic** `has_a_track-workshop-tutorial_topic` [Conference part -> Topic] | **has Topic** `hasTopic` [Working event -> Topic] *confTool* | 0.06 (0.27 / 0.06) | 1 / 2 | has Topic (0.06) | has tracks (0.08) | mutual-best | compound / partial overlap |
| conference-confOf | property | **has the last name** `has_the_last_name` [Person -> string] | **has Surname** `hasSurname` [Person -> string] *confTool* | 0.44 (0.13 / 0.44) | 1 / 1 | has Surname (0.44) | has the last name (0.44) | threshold | property naming |
| conference-edas | class | **Accepted contribution** `Accepted_contribution` [Reviewed contribution] | **Accepted Paper** `AcceptedPaper` [Paper] | 0.32 (0.38 / 0.32) | 2 / 1 | Accept Rating (0.32) | Accepted contribution (0.32) | scoring | compound / partial overlap |
| conference-edas | class | **Conference part** `Conference_part` [-] | **Conference Event** `ConferenceEvent` [-] | 0.27 (0.28 / 0.27) | 2 / 2 | Conference Session (0.28) | Conference proceedings (0.45) | scoring | compound / partial overlap |
| conference-edas | class | **Conference participant** `Conference_participant` [Person] | **Attendee** `Attendee` [Person] | 0.11 (0.05 / 0.11) | 2 / 2 | Conference Chair (0.17) | Invited speaker (0.12) | scoring | synonym / paraphrase |
| conference-edas | class | **Conference volume** `Conference_volume` [Conference] | **Conference** `Conference` [-] *Conference* | 0.00 (0.42 / 0.04) | - / - | Single Level Conference (0.11) | Conference (1.00) | cascade (Conference -> Conference) | compound / partial overlap |
| conference-edas | class | **Regular author** `Regular_author` [Conference contribution; Conference contributor; Contribution 1th author; Contribution co author] | **Author** `Author` [Person] | 0.25 (0.40 / 0.25) | 1 / 2 | Author (0.25) | Contribution co author (0.26) | mutual-best | head noun shared, modifier differs |
| conference-edas | class | **Rejected contribution** `Rejected_contribution` [Reviewed contribution] | **Rejected Paper** `RejectedPaper` [Paper] | 0.32 (0.38 / 0.32) | 2 / 1 | Reject Rating (0.38) | Rejected contribution (0.32) | scoring | compound / partial overlap |
| conference-edas | property | **has a review expertise** `has_a_review_expertise` [Submitted contribution -> Review expertise] | **has Rating** `hasRating` [Active Paper -> Review Rating] | 0.01 (0.15 / 0.01) | 5 / 7 | has Review History (0.15) | has a review (0.15) | scoring | property naming |
| conference-ekaw | class | **Accepted contribution** `Accepted_contribution` [Reviewed contribution] | **Accepted Paper** `Accepted_Paper` [Evaluated Paper] *Accepted paper to become part of proceedings.  Note that cam* | 0.23 (0.35 / 0.23) | 1 / 1 | Accepted Paper (0.23) | Accepted contribution (0.23) | threshold | compound / partial overlap |
| conference-ekaw | class | **Conference volume** `Conference_volume` [Conference] | **Conference** `Conference` [Scientific Event] *One edition of a conference* | 0.00 (0.36 / 0.10) | - / - | Conference Session (0.18) | Conference (0.83) | cascade (Conference -> Conference) | compound / partial overlap |
| conference-ekaw | class | **Conference www** `Conference_www` [Conference document] | **Web Site** `Web_Site` [Document] | 0.03 (0.04 / 0.03) | 4 / 1 | Conference Session (0.17) | Conference www (0.03) | scoring | synonym / paraphrase |
| conference-ekaw | class | **Early paid applicant** `Early_paid_applicant` [Paid applicant] | **Early Registered Participant** `Early-Registered_Participant` [Conference Participant] | 0.31 (0.21 / 0.31) | 1 / 1 | Early Registered Participant (0.31) | Early paid applicant (0.31) | threshold | compound / partial overlap |
| conference-ekaw | class | **Information for participants** `Information_for_participants` [Conference document] | **Programme Brochure** `Programme_Brochure` [Document] | 0.04 (0.03 / 0.04) | 1 / 2 | Programme Brochure (0.04) | Program committee (0.09) | mutual-best | synonym / paraphrase |
| conference-ekaw | class | **Late paid applicant** `Late_paid_applicant` [Paid applicant] | **Late Registered Participant** `Late-Registered_Participant` [Conference Participant] | 0.33 (0.21 / 0.33) | 1 / 1 | Late Registered Participant (0.33) | Late paid applicant (0.33) | threshold | compound / partial overlap |
| conference-ekaw | class | **Regular author** `Regular_author` [Conference contribution; Conference contributor; Contribution 1th author; Contribution co author] | **Paper Author** `Paper_Author` [Person] | 0.12 (0.30 / 0.12) | 3 / 1 | Regular Paper (0.15) | Regular author (0.12) | scoring | head noun shared, modifier differs |
| conference-ekaw | class | **Rejected contribution** `Rejected_contribution` [Reviewed contribution] | **Rejected Paper** `Rejected_Paper` [Evaluated Paper] | 0.41 (0.39 / 0.41) | 1 / 1 | Rejected Paper (0.41) | Rejected contribution (0.41) | threshold | compound / partial overlap |
| conference-ekaw | class | **Reviewed contribution** `Reviewed_contribution` [Submitted contribution] | **Evaluated Paper** `Evaluated_Paper` [Assigned Paper] | 0.01 (0.00 / 0.01) | 3 / 1 | Possible Reviewer (0.03) | Reviewed contribution (0.01) | scoring | synonym / paraphrase |
| conference-ekaw | class | **Submitted contribution** `Submitted_contribution` [Written contribution] | **Submitted Paper** `Submitted_Paper` [Paper] | 0.45 (0.38 / 0.45) | 1 / 1 | Submitted Paper (0.45) | Submitted contribution (0.45) | threshold | compound / partial overlap |
| conference-ekaw | property | **contributes** `contributes` [Person -> Conference document] | **author Of** `authorOf` [Person -> Document] | 0.00 (0.07 / 0.00) | - / - | written By (0.10) | has authors (0.38) | scoring | property naming |
| conference-iasted | class | **Active conference participant** `Active_conference_participant` [Conference contributor; Conference participant] | **Speaker** `Speaker` [Delegate] | 0.00 (0.00 / 0.00) | - / - | Conference activity (0.18) | Invited speaker (0.32) | scoring | synonym / paraphrase |
| conference-iasted | class | **Camera ready contribution** `Camera_ready_contribution` [Accepted contribution] | **Final manuscript** `Final_manuscript` [Submission] | 0.00 (0.00 / 0.00) | - / - | Camera ready manuscript deadline (0.37) | Extended abstract (0.01) | scoring | synonym / paraphrase |
| conference-iasted | class | **Conference part** `Conference_part` [-] | **Conference activity** `Conference_activity` [Activity] | 0.00 (0.27 / 0.00) | - / - | Conference days (0.00) | Conference participant (0.18) | scoring | compound / partial overlap |
| conference-iasted | class | **Conference proceedings** `Conference_proceedings` [-] | **Publication** `Publication` [Item] | 0.00 (0.00 / 0.00) | - / - | Conference activity (0.05) | Publisher (0.23) | scoring | synonym / paraphrase |
| conference-iasted | class | **Passive conference participant** `Passive_conference_participant` [Conference participant] | **Listener** `Listener` [Delegate] | 0.00 (0.00 / 0.00) | - / - | Conference activity (0.06) | Invited speaker (0.01) | scoring | synonym / paraphrase |
| conference-iasted | class | **Regular author** `Regular_author` [Conference contribution; Conference contributor; Contribution 1th author; Contribution co author] | **Author** `Author` [Speaker] | 0.21 (0.37 / 0.21) | 1 / 1 | Author (0.21) | Regular author (0.21) | threshold | head noun shared, modifier differs |
| conference-iasted | class | **Submitted contribution** `Submitted_contribution` [Written contribution] | **Submission** `Submission` [Document] | 0.38 (0.00 / 0.38) | 1 / 1 | Submission (0.38) | Submitted contribution (0.38) | threshold | synonym / paraphrase |
| conference-iasted | property | **contributes** `contributes` [Person -> Conference document] | **write** `write` [Person -> Item] | 0.08 (0.03 / 0.08) | 3 / 3 | prepare (0.17) | has authors (0.10) | scoring | property naming |
| conference-sigkdd | class | **Conference volume** `Conference_volume` [Conference] | **Conference** `Conference` [-] | 0.00 (0.42 / 0.01) | - / - | Conference hall (0.17) | Conference (1.00) | cascade (Conference -> Conference) | compound / partial overlap |
| conference-sigkdd | class | **Regular author** `Regular_author` [Conference contribution; Conference contributor; Contribution 1th author; Contribution co author] | **Author** `Author` [Speaker] | 0.25 (0.38 / 0.25) | 1 / 1 | Author (0.25) | Regular author (0.25) | threshold | head noun shared, modifier differs |
| conference-sigkdd | property | **gives presentations** `gives_presentations` [Active conference participant -> Presentation] | **presentation** `presentation` [Speaker -> Document] | 0.47 (0.40 / 0.47) | 1 / 1 | presentation (0.47) | gives presentations (0.47) | threshold | compound / partial overlap |
| conference-sigkdd | property | **is given by** `is_given_by` [Presentation -> Active conference participant] | **presentationed by** `presentationed_by` [Document -> Speaker] | 0.15 (0.19 / 0.15) | 1 / 2 | presentationed by (0.15) | gives presentations (0.41) | mutual-best | property naming |
| confOf-edas | class | **Event** `Event` [-] *confTool* | **Conference Event** `ConferenceEvent` [-] | 0.32 (0.41 / 0.32) | 1 / 1 | Conference Event (0.32) | Event (0.32) | threshold | head noun shared, modifier differs |
| confOf-edas | class | **Member PC** `Member_PC` [Person] *Reviewer* | **TPC Member** `TPCMember` [Person] | 0.31 (0.32 / 0.31) | 1 / 1 | TPC Member (0.31) | Member PC (0.31) | threshold | compound / partial overlap |
| confOf-edas | class | **Participant** `Participant` [Person] *confTool* | **Attendee** `Attendee` [Person] | 0.08 (0.04 / 0.08) | 1 / 1 | Attendee (0.08) | Participant (0.08) | threshold | synonym / paraphrase |
| confOf-edas | class | **Trip** `Trip` [Social event] | **Excursion** `Excursion` [Social Event] | 0.50 (0.10 / 0.50) | 1 / 1 | Excursion (0.50) | Trip (0.50) | threshold | synonym / paraphrase |
| confOf-edas | class | **Working event** `Working_event` [Event] | **Academic Event** `AcademicEvent` [Conference Event] | 0.07 (0.24 / 0.07) | 3 / 2 | Break Event (0.26) | Administrative event (0.20) | scoring | head noun shared, modifier differs |
| confOf-edas | property | **has Surname** `hasSurname` [Person -> string] *confTool* | **has Last Name** `hasLastName` [Person -> string] | 0.43 (0.17 / 0.43) | 1 / 1 | has Last Name (0.43) | has Surname (0.43) | threshold | property naming |
| confOf-edas | property | **reviewes** `reviewes` [Member PC -> Contribution] | **is Reviewing** `isReviewing` [Conference Chair, Reviewer, Session Chair, TPC Member -> Paper] | 0.28 (0.02 / 0.28) | 1 / 1 | is Reviewing (0.28) | reviewes (0.28) | threshold | property naming |
| confOf-edas | property | **writes** `writes` [Author -> Contribution] *confTool* | **has Related Paper** `hasRelatedPaper` [Author -> Paper] | 0.00 (0.03 / 0.00) | - / - | - | - | scoring | property naming |
| confOf-ekaw | class | **Contribution** `Contribution` [-] *confTool* | **Paper** `Paper` [Document] | 0.00 (0.00 / 0.00) | - / - | Contributed Talk (0.18) | Paper (0.84) | cascade (Paper -> Paper) | synonym / paraphrase |
| confOf-ekaw | class | **Participant** `Participant` [Person] *confTool* | **Conference Participant** `Conference_Participant` [Person] | 0.43 (0.50 / 0.43) | 1 / 1 | Conference Participant (0.43) | Participant (0.43) | threshold | head noun shared, modifier differs |
| confOf-ekaw | class | **Scholar** `Scholar` [Person] *price category - scholar* | **Student** `Student` [Person] | 0.00 (0.04 / 0.20) | - / - | Academic Institution (0.18) | Student (0.84) | cascade (Student -> Student) | synonym / paraphrase |
| confOf-ekaw | class | **Topic** `Topic` [-] *confTool* | **Research Topic** `Research_Topic` [-] | 0.37 (0.42 / 0.37) | 1 / 1 | Research Topic (0.37) | Topic (0.37) | threshold | head noun shared, modifier differs |
| confOf-ekaw | class | **Working event** `Working_event` [Event] | **Scientific Event** `Scientific_Event` [Event] | 0.31 (0.25 / 0.31) | 1 / 1 | Scientific Event (0.31) | Working event (0.31) | threshold | head noun shared, modifier differs |
| confOf-iasted | class | **Administrative event** `Administrative_event` [Event] | **Activity before conference** `Activity_before_conference` [Activity] | 0.00 (0.00 / 0.00) | - / - | Record of attendance (0.06) | Conference (0.08) | scoring | synonym / paraphrase |
| confOf-iasted | class | **Country** `Country` [-] | **State** `State` [-] | 0.32 (0.00 / 0.32) | 1 / 1 | State (0.32) | Country (0.32) | threshold | synonym / paraphrase |
| confOf-iasted | class | **Event** `Event` [-] *confTool* | **Activity** `Activity` [-] | 0.02 (0.00 / 0.02) | 2 / 2 | Item (0.07) | Organization (0.12) | scoring | synonym / paraphrase |
| confOf-iasted | class | **Reception** `Reception` [Social event] | **Coctail reception** `Coctail_reception` [Conference activity] | 0.15 (0.42 / 0.15) | 1 / 1 | Coctail reception (0.15) | Reception (0.15) | threshold | head noun shared, modifier differs |
| confOf-sigkdd | class | **Chair PC** `Chair_PC` [Person] *confTool* | **Program Chair** `Program_Chair` [Organizator] | 0.39 (0.29 / 0.39) | 1 / 1 | Program Chair (0.39) | Chair PC (0.39) | threshold | compound / partial overlap |
| confOf-sigkdd | class | **Member PC** `Member_PC` [Person] *Reviewer* | **Program Committee member** `Program_Committee_member` [Organizator] | 0.22 (0.21 / 0.22) | 1 / 1 | Program Committee member (0.22) | Member PC (0.22) | threshold | compound / partial overlap |
| confOf-sigkdd | property | **has Email** `hasEmail` [Person -> string] *confTool* | **E mail** `E-mail` [Person -> string] | 0.44 (0.03 / 0.44) | 1 / 1 | E mail (0.44) | has Email (0.44) | threshold | property naming |
| edas-ekaw | class | **Academic Event** `AcademicEvent` [Conference Event] | **Scientific Event** `Scientific_Event` [Event] | 0.37 (0.25 / 0.37) | 1 / 1 | Scientific Event (0.37) | Academic Event (0.37) | threshold | head noun shared, modifier differs |
| edas-ekaw | class | **Attendee** `Attendee` [Person] | **Conference Participant** `Conference_Participant` [Person] | 0.10 (0.05 / 0.10) | 2 / 2 | Invited Speaker (0.18) | Conference Chair (0.15) | scoring | synonym / paraphrase |
| edas-ekaw | class | **Conference Dinner** `ConferenceDinner` [Meal Event] | **Conference Banquet** `Conference_Banquet` [Social Event] | 0.48 (0.26 / 0.48) | 1 / 1 | Conference Banquet (0.48) | Conference Dinner (0.48) | threshold | compound / partial overlap |
| edas-ekaw | class | **Conference Event** `ConferenceEvent` [-] | **Event** `Event` [-] | 0.29 (0.49 / 0.29) | 1 / 1 | Event (0.29) | Conference Event (0.29) | threshold | head noun shared, modifier differs |
| edas-ekaw | class | **Topic** `Topic` [-] | **Research Topic** `Research_Topic` [-] | 0.44 (0.38 / 0.44) | 1 / 1 | Research Topic (0.44) | Topic (0.44) | threshold | head noun shared, modifier differs |
| edas-ekaw | property | **has Location** `hasLocation` [Conference Event -> Place] | **held In** `heldIn` [Event -> Location] | 0.16 (0.06 / 0.16) | 1 / 1 | held In (0.16) | has Location (0.16) | threshold | property naming |
| edas-ekaw | property | **is Reviewed By** `isReviewedBy` [Paper -> Conference Chair, Reviewer, Session Chair, TPC Member] | **has Reviewer** `hasReviewer` [Paper -> Possible Reviewer] | 0.21 (0.03 / 0.21) | 1 / 3 | has Reviewer (0.21) | has Rating (0.29) | mutual-best | property naming |
| edas-ekaw | property | **is Reviewing** `isReviewing` [Conference Chair, Reviewer, Session Chair, TPC Member -> Paper] | **reviewer Of Paper** `reviewerOfPaper` [Possible Reviewer -> Paper] | 0.21 (0.04 / 0.21) | 2 / 1 | has Reviewer (0.26) | is Reviewing (0.21) | scoring | property naming |
| edas-iasted | class | **Attendee** `Attendee` [Person] | **Delegate** `Delegate` [Conference building; Conference days; Person; Registration] | 0.00 (0.03 / 0.00) | - / - | Record of attendance (0.16) | Conference Chair (0.04) | scoring | synonym / paraphrase |
| edas-iasted | class | **Conference Dinner** `ConferenceDinner` [Meal Event] | **Dinner banquet** `Dinner_banquet` [Social program] | 0.43 (0.32 / 0.43) | 1 / 1 | Dinner banquet (0.43) | Conference Dinner (0.43) | threshold | compound / partial overlap |
| edas-iasted | class | **Conference Event** `ConferenceEvent` [-] | **Conference activity** `Conference_activity` [Activity] | 0.12 (0.29 / 0.12) | 1 / 1 | Conference activity (0.12) | Conference Event (0.12) | threshold | compound / partial overlap |
| edas-iasted | class | **Conference Venue Place** `ConferenceVenuePlace` [Place] | **Conference building** `Conference_building` [Building] | 0.27 (0.20 / 0.27) | 2 / 1 | Conference hall (0.30) | Conference Venue Place (0.27) | scoring | compound / partial overlap |
| edas-iasted | class | **Country** `Country` [-] | **State** `State` [-] | 0.30 (0.00 / 0.30) | 1 / 1 | State (0.30) | Country (0.30) | threshold | synonym / paraphrase |
| edas-iasted | class | **Dining Place** `DiningPlace` [Place] | **Conference restaurant** `Conference_restaurant` [Building] | 0.11 (0.00 / 0.11) | 4 / 3 | Hotel room (0.17) | Conference Dinner (0.31) | scoring | synonym / paraphrase |
| edas-iasted | class | **Paper** `Paper` [Document] | **Submission** `Submission` [Document] | 0.06 (0.07 / 0.06) | 5 / 1 | Form (0.16) | Paper (0.06) | scoring | synonym / paraphrase |
| edas-iasted | class | **Reception** `Reception` [Social Event] | **Coctail reception** `Coctail_reception` [Conference activity] | 0.13 (0.42 / 0.13) | 1 / 1 | Coctail reception (0.13) | Reception (0.13) | threshold | head noun shared, modifier differs |
| edas-iasted | class | **Slide Set** `SlideSet` [Document] | **Transparency** `Transparency` [Document] | 0.00 (0.05 / 0.00) | - / - | - | Paper (0.05) | scoring | synonym / paraphrase |
| edas-iasted | class | **Social Event** `SocialEvent` [Non Academic Event] | **Social program** `Social_program` [Conference activity] | 0.30 (0.31 / 0.30) | 1 / 1 | Social program (0.30) | Social Event (0.30) | threshold | compound / partial overlap |
| edas-iasted | class | **Welcome Talk** `WelcomeTalk` [Talk Event] | **Welcome address** `Welcome_address` [Conference activity] | 0.31 (0.30 / 0.31) | 1 / 1 | Welcome address (0.31) | Welcome Talk (0.31) | threshold | compound / partial overlap |
| edas-sigkdd | class | **Accommodation Place** `AccommodationPlace` [Place] | **Hotel** `Hotel` [Place] | 0.35 (0.07 / 0.35) | 1 / 1 | Hotel (0.35) | Accommodation Place (0.35) | threshold | synonym / paraphrase |
| edas-sigkdd | class | **Attendee** `Attendee` [Person] | **Listener** `Listener` [Person] | 0.14 (0.07 / 0.14) | 3 / 1 | Invited Speaker (0.23) | Attendee (0.14) | scoring | synonym / paraphrase |
| edas-sigkdd | class | **Conference Chair** `ConferenceChair` [Person] | **General Chair** `General_Chair` [Organizator] | 0.36 (0.31 / 0.36) | 1 / 1 | General Chair (0.36) | Conference Chair (0.36) | threshold | head noun shared, modifier differs |
| edas-sigkdd | class | **Conference Venue Place** `ConferenceVenuePlace` [Place] | **Conference hall** `Conference_hall` [Place] | 0.39 (0.25 / 0.39) | 1 / 1 | Conference hall (0.39) | Conference Venue Place (0.39) | threshold | compound / partial overlap |
| edas-sigkdd | property | **end Date** `endDate` [Conference -> dateTime] *Conference End Date and Time* | **End of conference** `End_of_conference` [Conference -> dateTime] | 0.47 (0.30 / 0.47) | 1 / 1 | End of conference (0.47) | end Date (0.47) | threshold | compound / partial overlap |
| edas-sigkdd | property | **has Cost Amount** `hasCostAmount` [Sponsorship -> int] | **Price** `Price` [Registration fee -> int] | 0.34 (0.00 / 0.34) | 1 / 1 | Price (0.34) | has Cost Amount (0.34) | threshold | property naming |
| edas-sigkdd | property | **has Name** `hasName` [Conference -> string] *Conference Full Name* | **Name of conference** `Name_of_conference` [Conference -> string] | 0.00 (0.32 / 0.20) | - / - | Name (0.51) | is Location Of (0.05) | cascade (has Name -> Name) | compound / partial overlap |
| edas-sigkdd | property | **start Date** `startDate` [Conference -> dateTime] *Conference Start Date and Time* | **Start of conference** `Start_of_conference` [Conference -> dateTime] | 0.45 (0.30 / 0.45) | 1 / 1 | Start of conference (0.45) | start Date (0.45) | threshold | compound / partial overlap |
| ekaw-iasted | class | **Event** `Event` [-] | **Activity** `Activity` [-] | 0.07 (0.00 / 0.07) | 2 / 2 | Item (0.10) | Organisation (0.08) | scoring | synonym / paraphrase |
| ekaw-iasted | class | **Paper Author** `Paper_Author` [Person] | **Author** `Author` [Speaker] | 0.31 (0.49 / 0.31) | 1 / 1 | Author (0.31) | Paper Author (0.31) | threshold | head noun shared, modifier differs |
| ekaw-sigkdd | class | **OC Member** `OC_Member` [Conference Participant] | **Organizing Committee member** `Organizing_Committee_member` [Organizator] | 0.15 (0.20 / 0.15) | 1 / 2 | Organizing Committee member (0.15) | Organising Agency (0.19) | mutual-best | head noun shared, modifier differs |
| ekaw-sigkdd | class | **PC Chair** `PC_Chair` [Conference Participant; PC Member] | **Program Chair** `Program_Chair` [Organizator] | 0.33 (0.26 / 0.33) | 1 / 1 | Program Chair (0.33) | PC Chair (0.33) | threshold | head noun shared, modifier differs |
| ekaw-sigkdd | class | **Paper Author** `Paper_Author` [Person] | **Author** `Author` [Speaker] | 0.00 (0.53 / 0.21) | - / - | Author of paper (0.64) | Multi author Volume (0.14) | cascade (Paper Author -> Author of paper) | head noun shared, modifier differs |
| iasted-sigkdd | class | **Deadline for notification of acceptance** `Deadline_for_notification_of_acceptance` [Deadline] | **Deadline Author notification** `Deadline_Author_notification` [Deadline] | 0.41 (0.33 / 0.41) | 1 / 1 | Deadline Author notification (0.41) | Deadline for notification of acceptance (0.41) | threshold | compound / partial overlap |
| iasted-sigkdd | class | **Sponsor** `Sponsor` [Person] | **Sponzor** `Sponzor` [-] | 0.00 (0.00 / 0.00) | - / - | Platinum Supporter (0.05) | Sponzorship (0.54) | scoring | synonym / paraphrase |

## False positives (all pairs)

| pair | kind | source | target | score (lex / emb) | blocks a gold pair? | category |
|---|---|---|---|---|---|---|
| cmt-conference | class | **Conference Chair** `ConferenceChair` [Chairman; Conference Member] | **Chair** `Chair` [Committee member] | 0.52 (0.52 / 0.39) | Chairman = Chair | head noun shared, modifier differs |
| cmt-conference | class | **Conference Member** `ConferenceMember` [Person] | **Conference participant** `Conference_participant` [Person] | 0.63 (0.31 / 0.63) | none (both entities unmatched in ra1) | compound / partial overlap |
| cmt-conference | class | **Paper** `Paper` [Document] | **Paper** `Paper` [Regular contribution] | 0.85 (0.85 / 0.77) | none (both entities unmatched in ra1) | identical label |
| cmt-conference | class | **Program Committee Member** `ProgramCommitteeMember` [Conference Member; Person] *The Program Committee consists of all members of the confere* | **Committee member** `Committee_member` [Person] | 0.55 (0.55 / 0.29) | none (both entities unmatched in ra1) | head noun shared, modifier differs |
| cmt-conference | class | **Reviewer** `Reviewer` [Conference Member; User] | **Reviewer** `Reviewer` [Person] | 0.84 (0.84 / 0.73) | none (both entities unmatched in ra1) | identical label |
| cmt-conference | property | **has Author** `hasAuthor` [Paper -> Author] | **has authors** `has_authors` [Conference document -> Person] | 0.87 (0.87 / 0.66) | none (both entities unmatched in ra1) | inflection / spelling |
| cmt-conference | property | **has Conference Member** `hasConferenceMember` [Conference -> Conference Member] | **has members** `has_members` [Committee -> Committee member] | 0.64 (0.64 / 0.36) | none (both entities unmatched in ra1) | compound / partial overlap |
| cmt-conference | property | **has Program Committee Member** `hasProgramCommitteeMember` [Program Committee -> Program Committee Member] | **has a program committee** `has_a_program_committee` [Conference volume -> Program committee] | 0.56 (0.56 / 0.56) | none (both entities unmatched in ra1) | compound / partial overlap |
| cmt-confOf | class | **Paper** `Paper` [Document] | **Paper** `Paper` [Contribution] *confTool* | 0.84 (0.84 / 0.65) | Paper = Contribution; Paper Full Version = Paper | identical label |
| cmt-confOf | property | **written By** `writtenBy` [Review -> Reviewer] | **written By** `writtenBy` [Contribution -> Author] *confTool* | 0.85 (0.85 / 0.64) | has Author = written By | identical label |
| cmt-edas | property | **date** `date` [Conference -> date] | **start Date** `startDate` [Conference -> dateTime] *Conference Start Date and Time* | 0.56 (0.46 / 0.56) | none (both entities unmatched in ra1) | compound / partial overlap |
| cmt-edas | property | **email** `email` [Person -> string] | **has Email** `hasEmail` [Contact Information, Person -> string] | 0.60 (0.60 / 0.65) | none (both entities unmatched in ra1) | compound / partial overlap |
| cmt-edas | property | **name** `name` [Conference, Person -> string] | **has Name** `hasName` [Conference -> string] *Conference Full Name* | 0.58 (0.58 / 0.56) | none (both entities unmatched in ra1) | compound / partial overlap |
| cmt-edas | property | **written By** `writtenBy` [Review -> Reviewer] | **is Written By** `isWrittenBy` [Paper -> Author] | 0.60 (0.60 / 0.56) | has Author = is Written By | inflection / spelling |
| cmt-ekaw | class | **Paper Abstract** `PaperAbstract` [Paper] | **Abstract** `Abstract` [Document] | 0.58 (0.58 / 0.51) | none (both entities unmatched in ra1) | head noun shared, modifier differs |
| cmt-ekaw | property | **write Review** `writeReview` [Reviewer -> Review] | **review Written By** `reviewWrittenBy` [Review -> Possible Reviewer] | 0.55 (0.24 / 0.55) | written By = review Written By | compound / partial overlap |
| cmt-ekaw | property | **written By** `writtenBy` [Review -> Reviewer] | **written By** `writtenBy` [Document -> Person] | 0.87 (0.87 / 0.70) | written By = review Written By | identical label |
| cmt-iasted | class | **Document** `Document` [-] | **Document** `Document` [Item] | 0.90 (0.90 / 0.60) | none (both entities unmatched in ra1) | identical label |
| cmt-sigkdd | class | **Paper Abstract** `PaperAbstract` [Paper] | **Abstract** `Abstract` [Document] | 0.57 (0.57 / 0.58) | none (both entities unmatched in ra1) | head noun shared, modifier differs |
| cmt-sigkdd | property | **date** `date` [Conference -> date] | **Date** `Date` [Deadline -> dateTime] | 0.87 (0.87 / 0.69) | none (both entities unmatched in ra1) | identical label |
| cmt-sigkdd | property | **name** `name` [Conference, Person -> string] | **Name** `Name` [Person -> string] | 0.93 (0.93 / 0.82) | none (both entities unmatched in ra1) | identical label |
| conference-confOf | class | **Camera ready contribution** `Camera_ready_contribution` [Accepted contribution] | **Camera Ready event** `Camera_Ready_event` [Administrative event] *confTool* | 0.50 (0.50 / 0.43) | none (both entities unmatched in ra1) | compound / partial overlap |
| conference-confOf | class | **Conference** `Conference` [-] | **Conference** `Conference` [Working event] | 0.87 (0.87 / 0.55) | Conference volume = Conference | identical label |
| conference-confOf | property | **reviews** `reviews` [Review -> Reviewed contribution] | **reviewes** `reviewes` [Member PC -> Contribution] | 0.50 (0.03 / 0.50) | none (both entities unmatched in ra1) | property naming |
| conference-edas | class | **Call for paper** `Call_for_paper` [Conference document] | **Call For Papers** `CallForPapers` [Call] *Call For Papers* | 0.92 (0.92 / 0.68) | none (both entities unmatched in ra1) | inflection / spelling |
| conference-edas | class | **Conference** `Conference` [-] | **Conference** `Conference` [-] *Conference* | 1.00 (1.00 / 0.89) | Conference volume = Conference | identical label |
| conference-edas | class | **Presentation** `Presentation` [Conference contribution] | **Paper Presentation** `PaperPresentation` [Academic Event] | 0.50 (0.50 / 0.41) | none (both entities unmatched in ra1) | head noun shared, modifier differs |
| conference-edas | property | **has a location** `has_a_location` [Conference volume -> string] | **has Location** `hasLocation` [Conference Event -> Place] | 0.82 (0.82 / 0.67) | none (both entities unmatched in ra1) | compound / partial overlap |
| conference-edas | property | **has a name** `has_a_name` [Conference, Conference part, Conference proceedings, Publisher, Topic -> string] | **has Name** `hasName` [Conference -> string] *Conference Full Name* | 0.63 (0.63 / 0.73) | none (both entities unmatched in ra1) | compound / partial overlap |
| conference-edas | property | **has an email** `has_an_email` [Person -> string] | **has Email** `hasEmail` [Contact Information, Person -> string] | 0.61 (0.61 / 0.86) | none (both entities unmatched in ra1) | compound / partial overlap |
| conference-edas | property | **has members** `has_members` [Committee -> Committee member] | **has Member** `hasMember` [Conference -> Person] | 0.88 (0.88 / 0.60) | none (both entities unmatched in ra1) | inflection / spelling |
| conference-edas | property | **was a member of** `was_a_member_of` [Committee member -> Committee] | **is Member Of** `isMemberOf` [Person -> Conference] | 0.55 (0.39 / 0.55) | none (both entities unmatched in ra1) | compound / partial overlap |
| conference-ekaw | class | **Camera ready contribution** `Camera_ready_contribution` [Accepted contribution] | **Camera Ready Paper** `Camera_Ready_Paper` [Paper] | 0.55 (0.55 / 0.45) | none (both entities unmatched in ra1) | compound / partial overlap |
| conference-ekaw | class | **Conference** `Conference` [-] | **Conference** `Conference` [Scientific Event] *One edition of a conference* | 0.83 (0.83 / 0.42) | Conference volume = Conference | identical label |
| conference-ekaw | class | **Conference participant** `Conference_participant` [Person] | **Conference Participant** `Conference_Participant` [Person] | 1.00 (1.00 / 1.00) | none (both entities unmatched in ra1) | identical label |
| conference-ekaw | class | **Invited talk** `Invited_talk` [Conference contribution; Presentation] | **Invited Talk** `Invited_Talk` [Individual Presentation] | 0.90 (0.90 / 0.87) | none (both entities unmatched in ra1) | identical label |
| conference-ekaw | class | **Paper** `Paper` [Regular contribution] | **Paper** `Paper` [Document] | 0.85 (0.85 / 0.74) | none (both entities unmatched in ra1) | identical label |
| conference-ekaw | class | **Reviewer** `Reviewer` [Person] | **Possible Reviewer** `Possible_Reviewer` [Person] | 0.61 (0.48 / 0.61) | none (both entities unmatched in ra1) | head noun shared, modifier differs |
| conference-ekaw | property | **has parts** `has_parts` [Conference volume -> Conference part] | **has Part** `hasPart` [? -> ?] | 0.90 (0.90 / 0.40) | none (both entities unmatched in ra1) | inflection / spelling |
| conference-iasted | class | **Presentation** `Presentation` [Conference contribution] | **Presentation** `Presentation` [Conference activity] | 0.90 (0.90 / 0.84) | none (both entities unmatched in ra1) | identical label |
| conference-iasted | property | **is given by** `is_given_by` [Presentation -> Active conference participant] | **is given by** `is_given_by` [Activity, Item -> Person] | 0.86 (0.86 / 0.50) | none (both entities unmatched in ra1) | identical label |
| conference-sigkdd | class | **Conference** `Conference` [-] | **Conference** `Conference` [-] | 1.00 (1.00 / 1.00) | Conference volume = Conference | identical label |
| confOf-edas | class | **Conference** `Conference` [Working event] | **Conference** `Conference` [-] *Conference* | 0.87 (0.87 / 0.55) | none (both entities unmatched in ra1) | identical label |
| confOf-edas | property | **has City** `hasCity` [Organization, Person -> City] *confTool* | **has City** `hasCity` [Contact Information -> string] | 0.85 (0.85 / 0.61) | none (both entities unmatched in ra1) | identical label |
| confOf-edas | property | **has Country** `hasCountry` [Organization, Person -> Country] *confTool* | **has Country** `hasCountry` [Conference -> Country] | 0.86 (0.86 / 0.76) | none (both entities unmatched in ra1) | identical label |
| confOf-edas | property | **has Email** `hasEmail` [Person -> string] *confTool* | **has Email** `hasEmail` [Contact Information, Person -> string] | 0.89 (0.89 / 0.84) | none (both entities unmatched in ra1) | identical label |
| confOf-edas | property | **has Phone** `hasPhone` [Person -> string] *confTool* | **has Phone** `hasPhone` [Contact Information -> string] | 0.86 (0.86 / 0.82) | none (both entities unmatched in ra1) | identical label |
| confOf-edas | property | **has Postal Code** `hasPostalCode` [Organization, Person -> string] *confTool* | **has Postal Code** `hasPostalCode` [Contact Information -> string] | 0.87 (0.87 / 0.74) | none (both entities unmatched in ra1) | identical label |
| confOf-edas | property | **has Street** `hasStreet` [Organization, Person -> string] *confTool* | **has Street** `hasStreet` [Contact Information -> string] | 0.85 (0.85 / 0.79) | none (both entities unmatched in ra1) | identical label |
| confOf-edas | property | **has Topic** `hasTopic` [Working event -> Topic] *confTool* | **has Topic** `hasTopic` [Conference, Paper -> Topic] | 0.85 (0.85 / 0.64) | none (both entities unmatched in ra1) | identical label |
| confOf-edas | property | **location** `location` [Social event, Working event -> string] *where the event is being held* | **has Location** `hasLocation` [Conference Event -> Place] | 0.56 (0.56 / 0.40) | none (both entities unmatched in ra1) | compound / partial overlap |
| confOf-ekaw | class | **Camera Ready event** `Camera_Ready_event` [Administrative event] *confTool* | **Camera Ready Paper** `Camera_Ready_Paper` [Paper] | 0.55 (0.55 / 0.44) | none (both entities unmatched in ra1) | compound / partial overlap |
| confOf-ekaw | class | **Paper** `Paper` [Contribution] *confTool* | **Paper** `Paper` [Document] | 0.84 (0.84 / 0.62) | Contribution = Paper | identical label |
| confOf-ekaw | class | **Regular** `Regular` [Participant] *confTool* | **Regular Paper** `Regular_Paper` [Paper] | 0.53 (0.53 / 0.32) | none (both entities unmatched in ra1) | compound / partial overlap |
| confOf-ekaw | property | **has Administrative Event** `hasAdministrativeEvent` [Working event -> Administrative event] | **has Event** `hasEvent` [Event -> Event] | 0.54 (0.54 / 0.22) | none (both entities unmatched in ra1) | compound / partial overlap |
| confOf-ekaw | property | **location** `location` [Social event, Working event -> string] *where the event is being held* | **location Of** `locationOf` [Location -> Event] | 0.50 (0.49 / 0.50) | none (both entities unmatched in ra1) | inflection / spelling |
| confOf-ekaw | property | **written By** `writtenBy` [Contribution -> Author] *confTool* | **written By** `writtenBy` [Document -> Person] | 0.85 (0.85 / 0.70) | none (both entities unmatched in ra1) | identical label |
| confOf-iasted | property | **writes** `writes` [Author -> Contribution] *confTool* | **write** `write` [Person -> Item] | 0.83 (0.83 / 0.53) | none (both entities unmatched in ra1) | inflection / spelling |
| edas-ekaw | class | **Conference** `Conference` [-] *Conference* | **Conference** `Conference` [Scientific Event] *One edition of a conference* | 0.83 (0.83 / 0.46) | none (both entities unmatched in ra1) | identical label |
| edas-ekaw | class | **Conference Session** `ConferenceSession` [-] | **Conference Session** `Conference_Session` [Session] | 1.00 (1.00 / 0.66) | none (both entities unmatched in ra1) | identical label |
| edas-ekaw | class | **Presenter** `Presenter` [Author] | **Presenter** `Presenter` [Conference Participant] | 0.85 (0.85 / 0.76) | none (both entities unmatched in ra1) | identical label |
| edas-ekaw | class | **Reviewer** `Reviewer` [Person] | **Possible Reviewer** `Possible_Reviewer` [Person] | 0.60 (0.50 / 0.60) | none (both entities unmatched in ra1) | head noun shared, modifier differs |
| edas-ekaw | property | **has Review History** `hasReviewHistory` [Reviewer -> Personal Review History] | **has Review** `hasReview` [Paper -> Review] | 0.54 (0.54 / 0.45) | none (both entities unmatched in ra1) | compound / partial overlap |
| edas-ekaw | property | **is Topic Of** `isTopicOf` [Topic -> Conference, Paper] | **topic Covered By** `topicCoveredBy` [Research Topic -> Event, Paper] | 0.53 (0.16 / 0.53) | none (both entities unmatched in ra1) | compound / partial overlap |
| edas-ekaw | property | **is Written By** `isWrittenBy` [Paper -> Author] | **written By** `writtenBy` [Document -> Person] | 0.61 (0.61 / 0.71) | none (both entities unmatched in ra1) | inflection / spelling |
| edas-iasted | class | **Conference Session** `ConferenceSession` [-] | **Session** `Session` [Lecture] | 0.52 (0.52 / 0.19) | none (both entities unmatched in ra1) | head noun shared, modifier differs |
| edas-iasted | class | **Person** `Person` [-] | **Person** `Person` [-] | 1.00 (1.00 / 1.00) | none (both entities unmatched in ra1) | identical label |
| edas-iasted | property | **is Written By** `isWrittenBy` [Paper -> Author] | **is writen by** `is_writen_by` [Item -> Person] | 0.78 (0.78 / 0.25) | none (both entities unmatched in ra1) | property naming |
| edas-sigkdd | property | **has Name** `hasName` [Conference -> string] *Conference Full Name* | **Name** `Name` [Person -> string] | 0.51 (0.51 / 0.33) | has Name = Name of conference | compound / partial overlap |
| ekaw-iasted | class | **Possible Reviewer** `Possible_Reviewer` [Person] | **Reviewer** `Reviewer` [Speaker] | 0.55 (0.42 / 0.55) | none (both entities unmatched in ra1) | head noun shared, modifier differs |
| ekaw-iasted | property | **held In** `heldIn` [Event -> Location] | **is held in** `is_held_in` [Activity -> Place, Time] | 0.64 (0.64 / 0.60) | none (both entities unmatched in ra1) | inflection / spelling |
| ekaw-iasted | property | **written By** `writtenBy` [Document -> Person] | **write** `write` [Person -> Item] | 0.52 (0.03 / 0.52) | none (both entities unmatched in ra1) | property naming |
| ekaw-sigkdd | class | **Paper Author** `Paper_Author` [Person] | **Author of paper** `Author_of_paper` [Author] | 0.64 (0.64 / 0.82) | Paper Author = Author | compound / partial overlap |
| iasted-sigkdd | class | **Document** `Document` [Item] | **Document** `Document` [-] | 0.90 (0.90 / 0.60) | none (both entities unmatched in ra1) | identical label |
| iasted-sigkdd | class | **Hotel room** `Hotel_room` [Place] | **Hotel** `Hotel` [Place] | 0.50 (0.50 / 0.82) | none (both entities unmatched in ra1) | compound / partial overlap |
| iasted-sigkdd | class | **Sponzorship** `Sponzorship` [Money] | **Sponzor** `Sponzor` [-] | 0.54 (0.00 / 0.54) | Sponsor = Sponzor | synonym / paraphrase |
| iasted-sigkdd | class | **Submissions deadline** `Submissions_deadline` [Deadline] | **Deadline Paper Submission** `Deadline_Paper_Submission` [Deadline] | 0.68 (0.68 / 0.68) | none (both entities unmatched in ra1) | compound / partial overlap |
| iasted-sigkdd | property | **is paid by** `is_paid_by` [Money -> Person] | **payed by** `payed_by` [Registration fee -> Person] | 0.54 (0.22 / 0.54) | none (both entities unmatched in ra1) | property naming |
| iasted-sigkdd | property | **obtain** `obtain` [Person -> Item] | **obtain** `obtain` [Author -> Award] | 0.84 (0.84 / 0.63) | none (both entities unmatched in ra1) | identical label |
| iasted-sigkdd | property | **pay** `pay` [Person -> Money] | **pay** `pay` [Person -> Registration fee] | 0.88 (0.88 / 0.76) | none (both entities unmatched in ra1) | identical label |

## Ablations (leave-one-pair-out cross-validation of the threshold)

Every variant is a cascade (lexical first, threshold 0.50) with CSLS-calibrated cosines on the leftovers.  For each
row the threshold 0.30-0.80 is the only tuned knob: *a-priori* is the fixed 0.50, *LOPO-CV* selects the threshold on
the other 20 pairs for each held-out pair (M3 micro-F1) and sums the held-out counts, *LOOO-CV* holds out all six
pairs of an ontology, and the last column is the best threshold on all 21 pairs (optimistic, for reference).
Encoders available in this run: `intfloat/e5-small-v2`, `BAAI/bge-small-en-v1.5`, `sentence-transformers/all-MiniLM-L6-v2`, `sentence-transformers/all-mpnet-base-v2`, `BAAI/bge-base-en-v1.5`, `intfloat/e5-base-v2`, `intfloat/e5-large-v2`, `BAAI/bge-large-en-v1.5`, `thenlper/gte-large`.

| family | variant | notes | a-priori @0.50 M1 P / R / F1 | M3 P / R / F1 | LOPO-CV M1 P / R / F1 | M3 P / R / F1 | LOOO-CV M3 F1 | best th on ra1 (optimistic) M1 / M3 |
|---|---|---|---|---|---|---|---|---|
| stage2 | stage2=embedding | e5-small, text=both | 0.82 / 0.68 / **0.744** | 0.71 / 0.63 / **0.668** | 0.77 / 0.67 / **0.718** | 0.65 / 0.63 / **0.641** | 0.653 | 0.43: 0.743 / 0.670 |
| stage2 | stage2=max | e5-small, text=both | 0.82 / 0.68 / **0.743** | 0.70 / 0.63 / **0.667** | 0.80 / 0.66 / **0.726** | 0.68 / 0.62 / **0.651** | 0.646 | 0.51: 0.742 / 0.668 |
| stage2 | stage2=mean | e5-small, text=both | 0.82 / 0.65 / **0.726** | 0.72 / 0.60 / **0.657** | 0.80 / 0.64 / **0.708** | 0.69 / 0.59 / **0.637** | 0.646 | 0.48: 0.730 / 0.660 |
| stage2 | stage2=noisy_or | e5-small, text=both | 0.69 / 0.79 / **0.736** | 0.57 / 0.74 / **0.646** | 0.68 / 0.78 / **0.727** | 0.57 / 0.72 / **0.638** | 0.642 | 0.50: 0.736 / 0.646 |
| text | text=label | e5-small, stage2=embedding | 0.81 / 0.66 / **0.726** | 0.68 / 0.62 / **0.645** | 0.81 / 0.64 / **0.717** | 0.68 / 0.60 / **0.638** | 0.632 | 0.52: 0.729 / 0.651 |
| text | text=both | e5-small, stage2=embedding | 0.82 / 0.68 / **0.744** | 0.71 / 0.63 / **0.668** | 0.77 / 0.67 / **0.718** | 0.65 / 0.63 / **0.641** | 0.653 | 0.43: 0.743 / 0.670 |
| text | text=full | e5-small, stage2=embedding | 0.79 / 0.69 / **0.737** | 0.68 / 0.65 / **0.662** | 0.79 / 0.68 / **0.729** | 0.67 / 0.64 / **0.653** | 0.656 | 0.51: 0.741 / 0.665 |
| text | text=rich | e5-small, stage2=embedding | 0.83 / 0.66 / **0.734** | 0.71 / 0.62 / **0.662** | 0.83 / 0.66 / **0.737** | 0.70 / 0.63 / **0.663** | 0.661 | 0.47: 0.737 / 0.665 |
| text | text=label+rich | e5-small, stage2=embedding | 0.84 / 0.65 / **0.730** | 0.72 / 0.61 / **0.658** | 0.78 / 0.66 / **0.717** | 0.64 / 0.63 / **0.638** | 0.648 | 0.43: 0.742 / 0.664 |
| text | text=label+head | e5-small, stage2=embedding | 0.70 / 0.71 / **0.708** | 0.57 / 0.66 / **0.615** | 0.76 / 0.64 / **0.697** | 0.63 / 0.59 / **0.609** | 0.609 | 0.59: 0.708 / 0.622 |
| text | text=label+full+head | e5-small, stage2=embedding | 0.75 / 0.69 / **0.716** | 0.62 / 0.65 / **0.633** | 0.78 / 0.67 / **0.724** | 0.65 / 0.63 / **0.639** | 0.635 | 0.52: 0.725 / 0.641 |
| text | text=label+rich+head | e5-small, stage2=embedding | 0.77 / 0.67 / **0.715** | 0.63 / 0.63 / **0.628** | 0.76 / 0.68 / **0.717** | 0.62 / 0.64 / **0.628** | 0.606 | 0.48: 0.725 / 0.634 |
| text | text=path | e5-small, stage2=embedding | 0.82 / 0.69 / **0.749** | 0.70 / 0.66 / **0.680** | 0.79 / 0.72 / **0.756** | 0.68 / 0.69 / **0.686** | 0.684 | 0.40: 0.764 / 0.693 |
| text+max | text=both, stage2=max | e5-small | 0.82 / 0.68 / **0.743** | 0.70 / 0.63 / **0.667** | 0.80 / 0.66 / **0.726** | 0.68 / 0.62 / **0.651** | 0.646 | 0.51: 0.742 / 0.668 |
| text+max | text=path, stage2=max | e5-small | 0.81 / 0.69 / **0.747** | 0.70 / 0.66 / **0.679** | 0.81 / 0.68 / **0.741** | 0.70 / 0.65 / **0.676** | 0.676 | 0.51: 0.751 / 0.684 |
| text+max | text=label+path, stage2=max | e5-small | 0.82 / 0.68 / **0.744** | 0.72 / 0.65 / **0.683** | 0.81 / 0.68 / **0.737** | 0.71 / 0.64 / **0.676** | 0.658 | 0.51: 0.746 / 0.684 |
| lexical | lexical head_weight=0.3 | e5-small, text=both | 0.73 / 0.73 / **0.728** | 0.61 / 0.67 / **0.640** | 0.76 / 0.69 / **0.722** | 0.66 / 0.64 / **0.646** | 0.640 | 0.56: 0.726 / 0.651 |
| lexical | lexical head_weight=0.5 | e5-small, text=both | 0.67 / 0.73 / **0.698** | 0.51 / 0.67 / **0.581** | 0.77 / 0.70 / **0.733** | 0.67 / 0.64 / **0.655** | 0.652 | 0.68: 0.733 / 0.655 |
| encoder | encoder=intfloat/e5-small-v2 | text=path, stage2=embedding | 0.82 / 0.69 / **0.749** | 0.70 / 0.66 / **0.680** | 0.79 / 0.72 / **0.756** | 0.68 / 0.69 / **0.686** | 0.684 | 0.40: 0.764 / 0.693 |
| encoder | encoder=BAAI/bge-small-en-v1.5 | text=path, stage2=embedding | 0.81 / 0.68 / **0.741** | 0.71 / 0.65 / **0.677** | 0.78 / 0.67 / **0.724** | 0.69 / 0.64 / **0.663** | 0.666 | 0.52: 0.744 / 0.683 |
| encoder | encoder=sentence-transformers/all-MiniLM-L6-v2 | text=path, stage2=embedding | 0.81 / 0.69 / **0.744** | 0.70 / 0.66 / **0.680** | 0.83 / 0.68 / **0.749** | 0.73 / 0.65 / **0.685** | 0.679 | 0.52: 0.749 / 0.685 |
| encoder | encoder=sentence-transformers/all-mpnet-base-v2 | text=path, stage2=embedding | 0.79 / 0.68 / **0.732** | 0.70 / 0.65 / **0.672** | 0.79 / 0.69 / **0.738** | 0.69 / 0.66 / **0.676** | 0.672 | 0.43: 0.743 / 0.682 |
| encoder | encoder=BAAI/bge-base-en-v1.5 | text=path, stage2=embedding | 0.82 / 0.67 / **0.739** | 0.71 / 0.64 / **0.676** | 0.84 / 0.66 / **0.740** | 0.73 / 0.63 / **0.676** | 0.669 | 0.52: 0.740 / 0.680 |
| encoder | encoder=intfloat/e5-base-v2 | text=path, stage2=embedding | 0.82 / 0.68 / **0.740** | 0.71 / 0.65 / **0.677** | 0.79 / 0.67 / **0.725** | 0.67 / 0.64 / **0.657** | 0.668 | 0.44: 0.751 / 0.680 |
| encoder | encoder=intfloat/e5-large-v2 | text=path, stage2=embedding | 0.82 / 0.67 / **0.738** | 0.72 / 0.64 / **0.679** | 0.83 / 0.69 / **0.751** | 0.72 / 0.66 / **0.687** | 0.680 | 0.45: 0.751 / 0.687 |
| encoder | encoder=BAAI/bge-large-en-v1.5 | text=path, stage2=embedding | 0.82 / 0.67 / **0.740** | 0.72 / 0.64 / **0.681** | 0.81 / 0.67 / **0.733** | 0.71 / 0.64 / **0.672** | 0.676 | 0.47: 0.745 / 0.684 |
| encoder | encoder=thenlper/gte-large | text=path, stage2=embedding | 0.82 / 0.66 / **0.728** | 0.72 / 0.63 / **0.672** | 0.79 / 0.64 / **0.711** | 0.69 / 0.62 / **0.654** | 0.659 | 0.52: 0.729 / 0.674 |
| ensemble | ensemble=small x3 | text=path, stage2=embedding | 0.82 / 0.68 / **0.745** | 0.71 / 0.65 / **0.680** | 0.80 / 0.69 / **0.742** | 0.68 / 0.66 / **0.671** | 0.665 | 0.49: 0.753 / 0.687 |
| ensemble | ensemble=base x3 | text=path, stage2=embedding | 0.82 / 0.67 / **0.736** | 0.72 / 0.64 / **0.676** | 0.79 / 0.71 / **0.748** | 0.68 / 0.68 / **0.675** | 0.670 | 0.36: 0.757 / 0.686 |
| ensemble | ensemble=large x3 | text=path, stage2=embedding | 0.83 / 0.66 / **0.738** | 0.73 / 0.64 / **0.679** | 0.84 / 0.65 / **0.736** | 0.74 / 0.63 / **0.677** | 0.672 | 0.52: 0.740 / 0.682 |
| ensemble | ensemble=e5-small + e5-large | text=path, stage2=embedding | 0.82 / 0.68 / **0.744** | 0.72 / 0.65 / **0.682** | 0.79 / 0.69 / **0.738** | 0.67 / 0.67 / **0.670** | 0.671 | 0.37: 0.756 / 0.686 |
| ensemble | ensemble=e5-large + bge-large | text=path, stage2=embedding | 0.82 / 0.66 / **0.735** | 0.72 / 0.64 / **0.677** | 0.82 / 0.65 / **0.725** | 0.72 / 0.63 / **0.669** | 0.664 | 0.47: 0.739 / 0.681 |
| ensemble | ensemble=all nine | text=path, stage2=embedding | 0.82 / 0.66 / **0.735** | 0.72 / 0.64 / **0.675** | 0.77 / 0.70 / **0.737** | 0.66 / 0.67 / **0.667** | 0.668 | 0.30: 0.754 / 0.683 |
| decoder | decoder=mutual-best |  | 0.82 / 0.69 / **0.749** | 0.70 / 0.66 / **0.680** | 0.79 / 0.72 / **0.756** | 0.68 / 0.69 / **0.686** | 0.684 | 0.40: 0.764 / 0.693 |
| decoder | decoder=hungarian |  | 0.82 / 0.69 / **0.749** | 0.70 / 0.66 / **0.680** | 0.79 / 0.70 / **0.746** | 0.67 / 0.67 / **0.673** | 0.678 | 0.42: 0.763 / 0.691 |
| decoder | decoder=mutual-best margin>=0.05 |  | 0.82 / 0.69 / **0.746** | 0.70 / 0.66 / **0.679** | 0.78 / 0.72 / **0.748** | 0.67 / 0.69 / **0.679** | 0.682 | 0.39: 0.762 / 0.695 |
| decoder | decoder=mutual-best margin>=0.1 |  | 0.81 / 0.68 / **0.741** | 0.71 / 0.65 / **0.678** | 0.77 / 0.73 / **0.751** | 0.68 / 0.70 / **0.689** | 0.687 | 0.31: 0.762 / 0.698 |
| decoder | decoder=mutual-best margin>=0.2 |  | 0.81 / 0.68 / **0.741** | 0.71 / 0.64 / **0.676** | 0.81 / 0.70 / **0.750** | 0.71 / 0.67 / **0.687** | 0.680 | 0.42: 0.754 / 0.690 |
| decoder | decoder=adaptive runner-up <= 0.5 x score |  | 0.57 / 0.78 / **0.656** | 0.45 / 0.75 / **0.565** | 0.56 / 0.79 / **0.657** | 0.45 / 0.76 / **0.564** | 0.564 | 0.39: 0.663 / 0.569 |
| decoder | decoder=adaptive runner-up <= 0.7 x score |  | 0.52 / 0.81 / **0.635** | 0.41 / 0.78 / **0.535** | 0.52 / 0.81 / **0.635** | 0.41 / 0.78 / **0.534** | 0.532 | 0.51: 0.635 / 0.536 |
| decoder | decoder=adaptive runner-up <= 0.9 x score |  | 0.45 / 0.82 / **0.581** | 0.34 / 0.78 / **0.478** | 0.45 / 0.82 / **0.581** | 0.34 / 0.78 / **0.477** | 0.477 | 0.48: 0.581 / 0.478 |
| decoder | decoder=adaptive score >= 0.3 x pair max | pair max is ~1.0 on every pair (exact matches), so this is a fixed threshold in disguise | 0.74 / 0.77 / **0.755** | 0.62 / 0.73 / **0.675** | 0.74 / 0.77 / **0.755** | 0.62 / 0.73 / **0.675** | 0.675 | 0.30: 0.755 / 0.675 |
| decoder | decoder=adaptive score >= 0.4 x pair max | pair max is ~1.0 on every pair (exact matches), so this is a fixed threshold in disguise | 0.80 / 0.73 / **0.766** | 0.69 / 0.70 / **0.694** | 0.80 / 0.73 / **0.766** | 0.69 / 0.70 / **0.694** | 0.694 | 0.40: 0.766 / 0.694 |
| decoder | decoder=adaptive score >= 0.5 x pair max | pair max is ~1.0 on every pair (exact matches), so this is a fixed threshold in disguise | 0.82 / 0.69 / **0.749** | 0.70 / 0.66 / **0.680** | 0.79 / 0.72 / **0.756** | 0.68 / 0.69 / **0.686** | 0.687 | 0.40: 0.764 / 0.693 |

### Union grids (the CV may pick any variant of the family, plus its threshold)

| union | points | LOPO-CV M1 P / R / F1 | M3 P / R / F1 | LOOO-CV M3 F1 | variants chosen (LOPO) | best on ra1 (optimistic) |
|---|---|---|---|---|---|---|
| stage2 x text | 918 | 0.79 / 0.72 / **0.756** | 0.68 / 0.69 / **0.686** | 0.682 | `text=path` | `text=path|th=0.40` |
| encoders | 459 | 0.79 / 0.71 / **0.747** | 0.68 / 0.68 / **0.680** | 0.674 | `encoder=intfloat/e5-large-v2`, `encoder=intfloat/e5-small-v2` | `encoder=intfloat/e5-small-v2|th=0.40` |
| ensembles | 306 | 0.78 / 0.68 / **0.727** | 0.67 / 0.65 / **0.658** | 0.663 | `ensemble=base x3`, `ensemble=e5-small + e5-large`, `ensemble=large x3`, `ensemble=small x3` | `ensemble=small x3|th=0.49` |
| decoders | 561 | 0.77 / 0.73 / **0.751** | 0.68 / 0.70 / **0.689** | 0.681 | `decoder=mutual-best margin>=0.1` | `decoder=mutual-best margin>=0.1|th=0.31` |
| everything | 2244 | 0.77 / 0.73 / **0.751** | 0.68 / 0.70 / **0.689** | 0.675 | `decoder=mutual-best margin>=0.1` | `decoder=mutual-best margin>=0.1|th=0.31` |

### Reading the ablations

- Baseline (e5-small, text=both, embedding-only second stage): LOPO-CV M1 0.718 / M3 0.641.
- Second-stage fusion: max 0.651, mean 0.637, noisy-or 0.638 (M3, LOPO-CV).
- Best single row under LOPO-CV: `decoder=adaptive score >= 0.4 x pair max` (pair max is ~1.0 on every pair (exact matches), so this is a fixed threshold in disguise) at M1 0.766 / M3 0.694; selected on the test set, so optimistic by itself.
- Union of every variant and threshold (2244 points), LOPO-CV: M1 0.751 / M3 0.689 (LOOO M3 0.675); a larger grid lets the selection overfit the 20 training pairs, which is why
  the union is not necessarily better than the best single family.
- What did not help (all within the 0.01-0.02 noise band of 21 pairs or clearly worse): the second-stage fusions,
  the head-noun field and the lexical head bonus (both admit the shared-head pairs ra1 rejects), the rich text
  (children / siblings / properties dilute the parent signal), every larger encoder and every ensemble, the
  Hungarian assignment and the runner-up margin (margin 0.1 is +0.003 M3, noise).  The `pair max` adaptive rule
  is a fixed threshold in disguise because every pair has an exact match scoring ~1.0, so its 0.4 row is the
  optimistic best-threshold number and not an honest gain.
- What helps: the path text, +0.045 M3 / +0.038 M1 under LOPO-CV over the baseline with the same encoder, and
  it is stable under leave-one-ontology-out.  Recall rises from 0.63 to 0.69 (M3) at unchanged precision.
