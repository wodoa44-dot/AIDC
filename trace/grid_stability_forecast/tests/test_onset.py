import unittest

from evaluate_onset import match_one_to_one


class OnsetMatchingTests(unittest.TestCase):
    def test_one_alarm_cannot_match_two_onsets(self):
        pairs, missed, false = match_one_to_one([1.0, 1.4], [0.8], 1.0)
        self.assertEqual(len(pairs), 1)
        self.assertEqual((missed, false), (1, 0))

    def test_post_onset_alarm_and_outside_lead_window_are_false(self):
        pairs, missed, false = match_one_to_one([2.0], [0.9, 2.0, 2.1], 1.0)
        self.assertEqual(pairs, [])
        self.assertEqual((missed, false), (1, 3))

    def test_earliest_eligible_alarm_is_selected(self):
        pairs, missed, false = match_one_to_one([2.0], [1.2, 1.8], 1.0)
        self.assertEqual(pairs, [(2.0, 1.2, 0.8)])
        self.assertEqual((missed, false), (0, 1))
