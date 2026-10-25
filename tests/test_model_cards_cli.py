"""`just fetch-cards NAME` with an unknown name prints the error and exits 2, like the other tools."""

from tools import model_cards


def test_an_unknown_name_is_an_error_line_not_a_traceback(capsys):
    assert model_cards.main(["nope"]) == 2
    assert "unknown service 'nope'; known: " in capsys.readouterr().err
