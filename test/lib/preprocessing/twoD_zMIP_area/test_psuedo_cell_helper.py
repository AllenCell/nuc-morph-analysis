from nuc_morph_analysis.lib.preprocessing.twoD_zMIP_area import pseudo_cell_helper, pseudo_cell_testing_helper

NUCLEUS_SIZE=pseudo_cell_testing_helper.NUCLEUS_SIZE
DISTANCE_BETWEEN_CELLS=pseudo_cell_testing_helper.DISTANCE_BETWEEN_CELLS
CELLS_PER_ROW=pseudo_cell_testing_helper.CELLS_PER_ROW

def test_get_pseudo_cell_boundaries():
    img = pseudo_cell_testing_helper.make_nucleus_image_array()
    try:
        df_2d = pseudo_cell_helper.get_pseudo_cell_boundaries(img)
        # assertions for nucleus features
        assert(df_2d['2d_area_nucleus'].values[0] == NUCLEUS_SIZE**2)
        assert(len(df_2d) == (CELLS_PER_ROW)**2)

        # assertions for pseudo cell features (after removal of outer edge cells)
        df_2d_dropna = df_2d.dropna(subset='2d_area_pseudo_cell') # remove all rows with NaN in the '2d_area_pseudo_cell' column
        assert df_2d_dropna.shape[0] == (CELLS_PER_ROW-2)**2 # the outer edge cells are NaN so new size should be (CELLS_PER_ROW-2)**2
        assert(df_2d_dropna['2d_area_pseudo_cell'].values[0] == (NUCLEUS_SIZE + DISTANCE_BETWEEN_CELLS)**2)
        assert(df_2d_dropna['2d_area_nuc_cell_ratio'].values[0] == (NUCLEUS_SIZE**2)/((NUCLEUS_SIZE + DISTANCE_BETWEEN_CELLS)**2))

    except Exception as e:
        print(f'Exception in testing get_pseudo_cell_boundaries: {e}')
        assert(False)

def test_get_pseudo_cell_boundaries_bonus_return_values():
    img = pseudo_cell_testing_helper.make_nucleus_image_array()
    try:
        df_2d, img_dict = pseudo_cell_helper.get_pseudo_cell_boundaries(
            labeled_nucleus_image=img,
            return_img_dict=True)
        assert(df_2d is not None)
        assert(img_dict is not None)  # are there more things to get checked here?

    except Exception as e:
        print(f'Exception in testing get_pseudo_cell_boundaries bonus return values: {e}')
        assert(False)