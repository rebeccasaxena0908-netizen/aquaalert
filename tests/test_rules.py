import pytest

from aquaalert.classify.base import classify_posts
from aquaalert.classify.rules import apply_rules, rescue_rule
from aquaalert.schemas import Label

OFFER, INFO, RESCUE = Label.OFFER_HELP, Label.INFO_UPDATE, Label.RESCUE_NEEDED


@pytest.mark.parametrize("text", [
    "Need rescue! 3 kids and 2 adults on the rooftop in Perumbavoor. Water is above 8 feet.",
    "Please help us, we are trapped on the first floor",
    "We are stuck at home, water entering the house, please send boats",
    "Help us!! family stranded near the bridge",
    "SOS from Aluva",
    "boats required at Kuttanad, 30 people on terrace",
    "Hum chhat par phase hue hain, bachao",
    "Mere ghar me paani ghus gaya, hum phans gaye hain",
    "madad chahiye, we are on our roof",
    "Rescue needed urgently at Edathua",
])
def test_distress_promoted_from_offer_and_info(text):
    for wrong in (OFFER, INFO):
        label, rule, conf = apply_rules(text, wrong)
        assert label == RESCUE and rule and 0 < conf <= 1, text


@pytest.mark.parametrize("text", [
    "We are rescuing people in Chengannur, boats available",
    "Volunteers ready with boats, DM us to coordinate rescue",
    "Boats available at Alappuzha for rescue operations",
    "Fishermen are bringing their boats to help rescue operations. Thank you heroes!",
    "We can rescue families stuck on rooftops, call us",
    "Our team of 25 NDRF volunteers reached Idukki, ready to assist.",
    "Rescue teams on the way to the trapped families",
    "Offering free stay for flood victims, message me",
    "Volunteers needed to help rescue trapped people, please share",
    "Doctors available for free medical aid at relief camps",
])
def test_genuine_offers_left_alone(text):
    label, rule, conf = apply_rules(text, OFFER)
    assert (label, rule, conf) == (OFFER, None, None), text


@pytest.mark.parametrize("text", [
    "Trains stuck in traffic near Shoranur station",
    "Heavy rain continues, water entering the dam reservoir",
    "Dam shutters opened, water level rising",
    "Hundreds trapped as dam opens, officials say",       # third-person news, no first-person marker
    "IMD issues red alert for Idukki",
])
def test_third_person_news_not_promoted(text):
    assert apply_rules(text, INFO)[0] == INFO, text


@pytest.mark.parametrize("text", [   # real false positives seen on HumAID: fundraising / volunteer appeals
    "Help us map impacts of Hurricane #Maria in our latest campaign",
    "We have raised $45,027 during our telethon. Follow link to help us raise more",
    "Help us support those affected by Hurricane Florence! Donate school supplies, clothing, toys",
    "Thank you for your support. This funding will help us provide urgent food relief to families",
    "SOS. National bloodbank is organising a voluntary blood donation. Needs donors at the metro station",
    "Please help us gather up-to-date info on Puerto Rico. Post in the FB group",
    "YOU helped to save 1000s of people stuck during the floods. REGISTER as a VOLUNTEER here",
    "Rescue team saved my 2 month old baby. Thanks to the rescue team",
])
def test_fundraising_and_gratitude_not_promoted(text):
    for label in (OFFER, INFO):
        assert apply_rules(text, label)[0] == label, text


@pytest.mark.parametrize("text", [   # narrations and rescuer wording seen on HumAID dev
    "I got stuck here after hurricane Maria. Conditions vary from bad to worse",
    "I was rescuing stranded people in my boat in Kerala when one man asked who I was",
    "Our Emergency Response Team's assignment is to rescue three dogs stranded in water",
    "We were trapped at the station overnight but we are fine now",
    "We rescued 40 families who were stuck on rooftops yesterday",
])
def test_past_tense_and_rescuer_wording_not_promoted(text):
    for label in (OFFER, INFO):
        assert apply_rules(text, label)[0] == label, text


def test_present_distress_still_detected_after_guards():
    assert apply_rules("We are stuck on our roof, water is entering the house", INFO)[0] == RESCUE
    assert apply_rules("Hum chhat par phase hue hain, bachao", OFFER)[0] == RESCUE
    assert apply_rules("We are stranded and the water is rising, please help us", OFFER)[0] == RESCUE


def test_unambiguous_requests_still_fire_inside_appeal_wording():
    assert apply_rules("We need rescue, please donate boats and share", OFFER)[0] == RESCUE
    assert apply_rules("Boats needed at Edathua, thank you volunteers", INFO)[0] == RESCUE


def test_only_offer_and_info_labels_are_overridden():
    text = "Please help us, we are trapped"
    for keep in (Label.NEED_SUPPLIES, Label.MISSING_PERSON, Label.INFRASTRUCTURE_DAMAGE, Label.IRRELEVANT, RESCUE):
        assert apply_rules(text, keep) == (keep, None, None)


def test_strong_cue_beats_offer_guard_but_contextual_does_not():
    assert rescue_rule("Volunteers ready, but we need rescue ourselves, help us")[0] == "distress_strong"
    assert rescue_rule("Volunteers ready to help, we are stuck on the roof") is None  # contextual + offer wording
    assert rescue_rule("we are stuck on the roof")[0] == "distress_contextual"


def test_classify_posts_records_rule(make_post):
    class FixedOffer:
        name = "fixed"

        def predict_scores(self, texts):
            return [{"offer_help": 0.9, "rescue_needed": 0.05, "info_update": 0.05} for _ in texts]

    posts = [make_post("Need rescue! stuck on the rooftop", "a"), make_post("Boats available, volunteers ready", "b")]
    a, b = classify_posts(posts, FixedOffer())
    assert (a.label, a.model_label, a.rule) == (RESCUE, OFFER, "distress_strong") and a.label_score == 0.8
    assert (b.label, b.model_label, b.rule) == (OFFER, None, None)
    plain = classify_posts(posts, FixedOffer(), rules=False)
    assert all(p.label == OFFER and p.rule is None for p in plain)
