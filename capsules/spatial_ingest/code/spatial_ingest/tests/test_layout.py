from layout import element_name, system_name


def test_bundle_names():
    assert element_name("cell_boundaries", "S1") == "S1_cell_boundaries"
    assert system_name("global", "S1") == "S1"
    assert system_name("aligned", "S1") == "S1_aligned"


def test_names_from_a_multi_sample_object_keep_one_prefix():
    assert element_name("S1_cell_boundaries", "S1") == "S1_cell_boundaries"
    assert element_name("cell_boundaries-a", "a", ("a",)) == "a_cell_boundaries"
    assert element_name("morphology-a", "P1", ("a",)) == "P1_morphology"
    assert system_name("global-a", "P1", ("a",)) == "P1"
    assert system_name("S1", "S1") == "S1"
    assert system_name("S1_aligned", "S1") == "S1_aligned"
