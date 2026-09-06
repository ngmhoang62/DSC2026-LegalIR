from exp_final.decomposition import inverse_document_frequency,query_views


def test_views_are_label_free_unique_and_bounded():
    text="Nhân viên sản xuất thực phẩm, nấu ăn nhà hàng được khám sức khỏe và điều trị bệnh nghề nghiệp như thế nào?"
    idf=inverse_document_frequency([text,"quy định bảo hiểm xã hội"])
    views=query_views(text,idf)
    assert 2<=len(views)<=3
    assert len(views)==len(set(v.casefold() for v in views))
    assert all(v.casefold()!=text.casefold() for v in views)
    assert "thực phẩm" in views[0]
    assert "bệnh nghề nghiệp" in views[1]


def test_short_query_has_safe_rare_view():
    text="Thủ tục thu hồi đất"
    views=query_views(text,inverse_document_frequency([text]))
    assert views==["quy định về Thủ tục thu hồi đất"]


def test_all_stopword_query_has_explicit_fallback():
    text="Là và của"
    assert query_views(text,inverse_document_frequency([text]))==["nội dung pháp lý: Là và của"]
