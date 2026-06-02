from scripts.summarize_prompt_injection_baseline import _rewrite_compliant


def test_dotted_field_compliance_allows_old_substring_inside_new_name() -> None:
    code = "mapping = map(self.add_ten_updated, seq)"
    assert _rewrite_compliant(code, {".add_ten": ".add_ten_updated"}) is True


def test_identifier_compliance_is_boundary_aware() -> None:
    assert _rewrite_compliant("account = get_account()", {"user": "account"}) is True
    assert _rewrite_compliant("user_id = account.id", {"user": "account"}) is True
    assert _rewrite_compliant("user = account", {"user": "account"}) is False
