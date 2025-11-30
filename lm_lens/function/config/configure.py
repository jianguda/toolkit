class Configure:
    def __init__(self, scale, data_name, model_name, subtask_code=None):
        self.scale = scale
        self.data_name = data_name
        self.model_name = model_name

        # ========== ABLATION STUDY CONFIGURATION ==========
        self.ABL0_APPROACH = 'me-iter'
        # for RQ2
        self.ABL1_ABLATION_LAYER = 'all'
        # for RQ2
        self.ABL2_ABLATION_RATIO = 1.0  # 1/1 1/4 1/16 1/64 1/256
        self.ABL2_ABLATION_WHICH = 'none'  # autoly evenly layerly maskmax maskmin none

        # actv: locate neurons by activation
        # * attr: locate neurons by attribution
        # rand: locate neurons randomly
        self.ABL1_LOCATING_REFERENCE = 'attr'

        # basis: use the semantic basis of the target token
        # * delta: use the semantic bases of (target - argmax)
        self.ABL2_ESTIMATING_SEMANTIC = 'delta'

        # * coeff: the coefficient is adaptive
        # plain: the coefficient is always one
        self.ABL2_ESTIMATING_SCALE_BY = 'coeff'

        # * gain: gain-guided, by reranking via simulated editing
        # score: score-guided, directly using attribution scores
        self.ABL3_PLANNING_RERANK_BY = 'gain'

        # refresh
        if subtask_code is not None:
            self._refresh(subtask_code)

    def _refresh(self, subtask_code: str):
        subtask_details = subtask_code.split('.')
        if len(subtask_details) != 3:
            return None
        subtask_type, subtask_key, subtask_value = subtask_details

        assert subtask_type in ['approach', 'locating', 'estimating', 'planning']
        if subtask_type == 'approach':
            assert subtask_key in ['0', '1', '2']
            if subtask_key == '0':
                assert subtask_value in ['finetune', 'mint', 'me-sgd', 'me-iter', 'me-batch']
                self.ABL0_APPROACH = subtask_value
            if subtask_key == '1':
                assert subtask_value in ['one', 'all', '1quarter', '2quarter', '3quarter']
                self.ABL1_ABLATION_LAYER = subtask_value
            if subtask_key == '2':
                which, ratio = subtask_value.split('-')
                self.ABL2_ABLATION_WHICH = str(which)
                self.ABL2_ABLATION_RATIO /= int(ratio)
        if subtask_type == 'locating':
            assert subtask_key in ['1']
            if subtask_key == '1':
                assert subtask_value in ['actv', 'attr', 'rand']
                assert subtask_value not in ['attr']
                self.ABL1_LOCATING_REFERENCE = subtask_value
        # TODO update the variants...
        if subtask_type == 'estimating':
            assert subtask_key in ['1', '2']
            if subtask_key == '1':
                assert subtask_value in ['basis', 'delta']
                assert subtask_value not in ['delta']
                self.ABL2_ESTIMATING_SEMANTIC = subtask_value
            if subtask_key == '2':
                assert subtask_value in ['coeff', 'plain']
                assert subtask_value not in ['coeff']
                self.ABL2_ESTIMATING_SCALE_BY = subtask_value
        if subtask_type == 'planning':
            assert subtask_key in ['1', '2', '3']
            # if subtask_key == '1':
            #     assert subtask_value in ['prob', 'rank', 'blend']
            #     assert subtask_value not in ['blend']
            #     self.ABL3_PLANNING_OBJECTIVE = subtask_value
            if subtask_key == '2':
                assert subtask_value in ['gain', 'score']
                assert subtask_value not in ['gain']
                self.ABL3_PLANNING_RERANK_BY = subtask_value
            # if subtask_key == '3':
            #     assert subtask_value in ['greedy', 'steady']
            #     assert subtask_value not in ['greedy']
            #     self.ABL3_PLANNING_PACE = subtask_value
