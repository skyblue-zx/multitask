# Code of Paper 
## Title: multitask_learning_framework_for_spoken_language_assessment

- The path "model_core/src/modles" is the directory of models defination.
- The model list:
   - T-Rand:
      - No Multi-task: text_single_model_no_pretrain.py
      - Multi-task: text_single_model_no_pretrain_embedding_share_and_gate_merge.py
   - A-PAA:
      - No Multi-task: audio_single_model_no_pretrain.py
      - Multi-task: audio_single_model_no_pretrain_gate_merge.py
   - TA-DiDi:
      - No Multi-task: didi_multimodel.py
      - Multi-task: didi_multimodel_embedding_share_and_output_merge.py
   - T-Bert:
      - No Multi-task: text_single_model_bert.py
      - Multi-task: text_single_model_and_text_gate_merge.py
   - A-VGGish:
      - No Multi-task: audio_single_model_lstm.py
      - Multi-task: audio_single_model_lstm_text_gate_merge.py
   - TA-BV:
      - No Multi-task: didi_multimodel_bert_vggish.py
      - Multi-task: didi_multimodel_bert_vggish_and_embedding_share_and_text_gate_merge.py
