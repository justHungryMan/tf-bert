"""Bert(wikipedia_en + bookcorpus) dataset."""

import tensorflow_datasets as tfds
import os
import math
import random

from tqdm import tqdm

import tensorflow.compat.v2 as tf
import sys



_DESCRIPTION = """
Bert(wikipedia_en + bookcorpus) Dataset
Code was implemented by Jun (21.12.28)
"""

_CITATION = """
Wikiepedia_en + bookcorpus from huggingface
"""
_TOKENIZER_JSON_PATH = './tokenizer.json'
_VOCAB_SIZE = 30_522
_DUPE_FACOTR = 1
_MAX_SEQUENCE_LENGTH = 128
_MAX_PREDICTIONS_PER_SEQ = 20
_MASKED_LM_PROB = 0.15


class BertBeamDataset(tfds.core.GeneratorBasedBuilder):
    # DatasetBuilder for Bert dataset.
    VERSION = tfds.core.Version('1.0.0')
    RELEASE_NOTES = {
        '1.0.0': 'Bert(wikipedia_en + bookcorpus'
    }


    def _info(self) -> tfds.core.DatasetInfo:

        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict({
                'input_ids': tfds.features.Sequence(tf.int64, _MAX_SEQUENCE_LENGTH),
                'input_mask': tfds.features.Sequence(tf.int64, _MAX_SEQUENCE_LENGTH),
                'segment_ids': tfds.features.Sequence(tf.int64, _MAX_SEQUENCE_LENGTH),
                'masked_lm_positions': tfds.features.Sequence(tf.int64, _MAX_PREDICTIONS_PER_SEQ),
                'masked_lm_ids': tfds.features.Sequence(tf.int64, _MAX_PREDICTIONS_PER_SEQ),
                'masked_lm_weights': tfds.features.Sequence(tf.float32, _MAX_PREDICTIONS_PER_SEQ),
                'next_sentence_labels': tfds.features.Sequence(tf.bool, 1),
            }),
    
            # supervised_keys=('image', 'label'), 
            homepage='https://huggingface.co/docs/datasets/index.html',
            citation=_CITATION,
        )

    
    def _split_generators(self, dl_manager):
        import datasets
        import tokenizers
        
        bookcorpus_dataset = datasets.load_dataset(
            'bookcorpus'
        )

        # wiki_dataset = datasets.load_dataset(
        #     'wikipedia', '20200501.en'
        # )
        # wiki_dataset = wiki_dataset.remove_columns("title")

        # assert bookcorpus_dataset['train'].features.type == wiki_dataset['train'].features.type
        bert_dataset = bookcorpus_dataset['train']
        bert_dataset = bert_dataset[:204800]
        # bert_dataset = datasets.concatenate_datasets([bookcorpus_dataset['train'], wiki_dataset['train']])

        if not os.path.isfile(_TOKENIZER_JSON_PATH):
            print(f"Can not find pretrained tokenizer.")
            print(f"Train Start..")

            tokenizer = tokenizers.Tokenizer(tokenizers.models.WordPiece(unk_known="[UNK]"))
            tokenizer.normalizer = tokenizers.normalizers.Sequence([
                tokenizers.normalizers.NFD(),
                tokenizers.normalizers.Lowercase(),
                tokenizers.normalizers.StripAccents()
                ])
            tokenizer.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()

            trainer = tokenizers.trainers.WordPieceTrainer(
                vocab_size=_VOCAB_SIZE,
                special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]
            )

            def batch_iterator(dataset):
                batch_length = 1024
                for i in range(0, len(dataset), batch_length):
                    yield dataset[i: i + batch_length]['text']
            tokenizer.train_from_iterator(batch_iterator(dataset=bert_dataset), trainer=trainer, length=len(bert_dataset))
            tokenizer.save(_TOKENIZER_JSON_PATH)

            print(f"Finish training tokenizer")

        print(f"Load Tokenizer")
        tokenizer = tokenizers.Tokenizer.from_file(_TOKENIZER_JSON_PATH)
        tokenizer.enable_truncation(
            max_length=_MAX_SEQUENCE_LENGTH
        )
        tokenizer.enable_padding(
            pad_id=3,
            pad_token="[PAD]",
            pad_to_multiple_of=_MAX_SEQUENCE_LENGTH
        )
        tokenizer.post_processor = tokenizers.processors.TemplateProcessing(
            single="[CLS] $A [SEP]",
            pair="[CLS] $A [SEP] $B:1 [SEP]:1",
            special_tokens=[
                ("[CLS]", tokenizer.token_to_id("[CLS]")),
                ("[SEP]", tokenizer.token_to_id("[SEP]"))
            ]
        )

        print(f"Duplicate dataset: {len(bert_dataset) * _DUPE_FACOTR}")
        # bert_dataset = datasets.concatenate_datasets([bert_dataset] * _DUPE_FACOTR)

        print(f"Generate..")
        return {
            'train': self._generate_examples(ds=bert_dataset, tokenizer=tokenizer),
            # 'validation': self._generate_examples(paths=data['validation'], synsets=synsets)
        }

    def _generate_examples(self, ds, tokenizer):
        random.seed(22)

        max_len = len(ds)

        def _process_example(ds):
            # https://d2l.ai/chapter_natural-language-processing-pretraining/bert-dataset.html
            def _replace_mlm_tokens(self, tokens, candidate_pred_positions, num_mlm_preds, tokenizer):
                mlm_input_tokens = [token for token in tokens]
                pred_positions_and_labels = []
                random.shuffle(candidate_pred_positions)

                for mlm_pred_position in candidate_pred_positions:
                    if len(pred_positions_and_labels) >= num_mlm_preds:
                        break
                    masked_token = None

                    if random.random() < 0.8:
                        masked_token = '[MASK]'
                    else:
                        if random.random() < 0.5:
                            masked_token = tokens[mlm_pred_position]
                        else:
                            # Except [UNK], [CLS], [SEP], [PAD], [MASK]
                            rand_token_id = random.randint(5, _VOCAB_SIZE)
                            masked_token = tokenizer.id_to_token(rand_token_id)
                    mlm_input_tokens[mlm_pred_position] = masked_token
                    pred_positions_and_labels.append((mlm_pred_position, tokens[mlm_pred_position]))

                return mlm_input_tokens, pred_positions_and_labels

            text = ds[i]['text']
            
            next_sentence_prob = random.random()
            if next_sentence_prob < 0.5 and i != max_len - 1:
                next_text = ds[i + 1]['text']
                next_sentence_labels = 1
            else:
                next_text = ds[random.sample(range(max_len), 1)[0]]['text']
                next_sentence_labels = 0

            tokenized_text = tokenizer.encode(text, next_text)


            input_ids = tokenized_text.ids
            input_mask = tokenized_text.attention_mask
            segment_ids = tokenized_text.type_ids

            # mlm_tokens
            candidate_pred_positions = [i for i, x in enumerate(tokenized_text.special_tokens_mask) if x == 0]
            mlm_input_tokens, pred_positions_and_labels = self._replace_mlm_tokens(
                tokens=input_ids,
                candidate_pred_positions=candidate_pred_positions,
                num_mlm_preds=min(max(1, round(len(candidate_pred_positions) * 0.15)), _MAX_PREDICTIONS_PER_SEQ),
                tokenizer=tokenizer,
            )
            pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
            mlm_pred_positions = [v[0] for v in pred_positions_and_labels]
            mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
            mlm_weights = [1.0] * len(mlm_pred_labels)
            
            while len(mlm_pred_positions) < _MAX_PREDICTIONS_PER_SEQ:
                mlm_pred_positions.append(0)
                # id of [PAD] 
                mlm_pred_labels.append(3)
                mlm_weights.append(0.0)
            record = {
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'masked_lm_positions': mlm_pred_positions,
                'masked_lm_ids': mlm_pred_labels,   
                'masked_lm_weights': mlm_weights,
                'next_sentence_labels': [next_sentence_labels]
            }
            return text, record

        beam = tfds.core.lazy_imports.apache_beam
        return (
            beam.Create(ds) 
            | beam.Map(_process_example)
        )
        # for i, data in tqdm(enumerate(ds), total=max_len):
        #     text = ds[i]['text']
            
        #     next_sentence_prob = random.random()
        #     if next_sentence_prob < 0.5 and i != max_len - 1:
        #         next_text = ds[i + 1]['text']
        #         next_sentence_labels = 1
        #     else:
        #         next_text = ds[random.sample(range(max_len), 1)[0]]['text']
        #         next_sentence_labels = 0

        #     tokenized_text = tokenizer.encode(text, next_text)


        #     input_ids = tokenized_text.ids
        #     input_mask = tokenized_text.attention_mask
        #     segment_ids = tokenized_text.type_ids

        #     # mlm_tokens
        #     candidate_pred_positions = [i for i, x in enumerate(tokenized_text.special_tokens_mask) if x == 0]
        #     mlm_input_tokens, pred_positions_and_labels = self._replace_mlm_tokens(
        #         tokens=input_ids,
        #         candidate_pred_positions=candidate_pred_positions,
        #         num_mlm_preds=min(max(1, round(len(candidate_pred_positions) * 0.15)), _MAX_PREDICTIONS_PER_SEQ),
        #         tokenizer=tokenizer,
        #     )
        #     pred_positions_and_labels = sorted(pred_positions_and_labels, key=lambda x: x[0])
        #     mlm_pred_positions = [v[0] for v in pred_positions_and_labels]
        #     mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
        #     mlm_weights = [1.0] * len(mlm_pred_labels)
            
        #     while len(mlm_pred_positions) < _MAX_PREDICTIONS_PER_SEQ:
        #         mlm_pred_positions.append(0)
        #         # id of [PAD] 
        #         mlm_pred_labels.append(3)
        #         mlm_weights.append(0.0)
        #     # print('mlm_input_tokens: ', mlm_input_tokens)
        #     # print('pred_positions_and_labels: ', pred_positions_and_labels)

        #     # print('mlm_pred_positions', mlm_pred_positions)
        #     # print('mlm_pred_labels', mlm_pred_labels)
        #     # print('mlm_weights:', mlm_weights)

        #     # print('--')
        #     # print(len(input_ids))

        #     # print(text)
        #     # print(tokenized_text.tokens)
        #     # print('token_id    :', input_ids)
        #     # print('input_mask  :', input_mask)
        #     # print('segment_ids :', segment_ids)
        #     # print('sepcial_mask:', tokenized_text.special_tokens_mask)

        #     try:
        #         record = {
        #             'input_ids': input_ids,
        #             'input_mask': input_mask,
        #             'segment_ids': segment_ids,
        #             'masked_lm_positions': mlm_pred_positions,
        #             'masked_lm_ids': mlm_pred_labels,
        #             'masked_lm_weights': mlm_weights,
        #             'next_sentence_labels': [next_sentence_labels]
        #         }

        #         yield text, record
        #     except Exception as e:
        #         continue

    
         

        


# if __name__ == '__main__':
#     # test = Bert()
#     tfds.enable_progress_bar()

#     # if not os.path.exists('gs://justhungryman/tfds'):
#     #     os.makedirs('./data')
#     flags = ['--runner=DataflowRunner', '--project=justhungryman', '--job_name=bert-gen', '--staging_location=gs://justhungryman/binaries', '--temp_location=gs://justhungryman/temp', 'requirements_file=/tmp/beam_requirements.txt']
#     beam = tfds.core.lazy_imports.apache_beam
#     dl_config = tfds.download.DownloadConfig(
#         beam_options=beam.options.pipeline_options.PipelineOptions(flags=flags, pipeline_type_check=False)
#     )

#     builder = tfds.builder('jun', data_dir='gs://justhungryman/tfds', try_gcs=False)
#     builder.download_and_prepare(
#         download_dir='gs://justhungryman/tfds',
#         download_config=dl_config
#     )
#     ds = builder.as_dataset(split='train')
#     print(builder.info)
#     print(builder.info.splits['train'].num_examples)

