import unittest


PARTICIPANT_ROOT = __file__.replace("\\", "/").rsplit("/", 2)[0]
SERVER_ROOT = PARTICIPANT_ROOT.rsplit("/", 1)[0] + "/2026-HAIC/simulator"
SHARED_FILES = (
    "core/track_variables.py",
    "core/obstacle_contacts.py",
    "core/vendor/car_racing.py",
    "core/vendor/car_dynamics.py",
)


def server_repository_available():
    try:
        with open(SERVER_ROOT + "/core/track_variables.py", "rb"):
            return True
    except OSError:
        return False


@unittest.skipUnless(server_repository_available(), "server repository is not available")
class TestServerParity(unittest.TestCase):
    def test_shared_environment_files_match_server(self):
        for relative_path in SHARED_FILES:
            with self.subTest(path=relative_path):
                with open(PARTICIPANT_ROOT + "/" + relative_path, "rb") as participant:
                    participant_content = participant.read()
                with open(SERVER_ROOT + "/" + relative_path, "rb") as server:
                    server_content = server.read()
                self.assertEqual(
                    participant_content,
                    server_content,
                    f"participant file differs from server: {relative_path}",
                )


if __name__ == "__main__":
    unittest.main()
