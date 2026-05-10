from legal_intel.runtime.embedding_similarity import cosine_similarity_uvec


def test_cosine_identical_unit_vectors():
    a = [1.0, 0.0, 0.0]
    b = [1.0, 0.0, 0.0]
    assert abs(cosine_similarity_uvec(a, b) - 1.0) < 1e-9


def test_cosine_orthogonal():
    a = [1.0, 0.0]
    b = [0.0, 1.0]
    assert abs(cosine_similarity_uvec(a, b)) < 1e-9
