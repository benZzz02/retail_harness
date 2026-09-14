from __future__ import annotations

import unittest

from refundpilot.user_simulator import CodexCliUserSimulator


class CodexCliUserSimulatorTest(unittest.TestCase):
    def _simulator(self, instruction: str) -> CodexCliUserSimulator:
        simulator = CodexCliUserSimulator(executable="/usr/bin/true")
        simulator._instruction = instruction
        return simulator

    def test_derives_name_from_benchmark_account_id(self) -> None:
        simulator = self._simulator(
            "You are mei_kovacs_8020 (zip code 28236) and need an exchange."
        )

        self.assertEqual(
            simulator._safe_identity_derivation(),
            {
                "first_name": "Mei",
                "last_name": "Kovacs",
                "account_user_id": "mei_kovacs_8020",
                "zip": "28236",
            },
        )

    def test_auth_prompt_uses_name_instead_of_inventing_email(self) -> None:
        simulator = self._simulator(
            "You are mei_kovacs_8020 (zip code 28236) and need an exchange."
        )

        response = simulator._grounded_auth_response(
            "Please provide your email address, or your first and last name "
            "with ZIP code."
        )

        self.assertEqual(
            response,
            "I don't have the email address handy. My name is Mei Kovacs, "
            "and my ZIP code is 28236.",
        )
        self.assertNotIn("@", response)

    def test_auth_prompt_uses_stated_email_when_available(self) -> None:
        simulator = self._simulator(
            "You are mia_garcia_4516 (mia.garcia2723@example.com)."
        )

        self.assertEqual(
            simulator._grounded_auth_response("Please provide your email."),
            "The email address is mia.garcia2723@example.com.",
        )


if __name__ == "__main__":
    unittest.main()
