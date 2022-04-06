import argparse
import logging
import os
from typing import Any, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from overrides import overrides
from torch.utils.data import DataLoader, TensorDataset
from transformers import PreTrainedTokenizer

from baseline.data.base import DataProcessor, KlueDataModule

import re

logger = logging.getLogger(__name__)
max_seq_length2 = 510

p1 = re.compile("를$")
p2 = re.compile("가$")
p3 = re.compile("는$")
p4 = re.compile("과$")
p5 = re.compile("에게는$")
p6 = re.compile("고$")


class KlueDPInputExample:
    """A single training/test example for Dependency Parsing in .conllu format

    Args:
        guid : Unique id for the example
        text : string. the original form of sentence
        token_id : token id
        token : 어절
        pos : POS tag(s)
        head : dependency head
        dep : dependency relation
    """

    def __init__(
            self, guid: str, text: str, sent_id: int, token_id: int, token: str, lemma: str, pos: str, pos2: str,
            pos3: str, pos4: str, pos5: str,
            head: str, dep: str
    ) -> None:
        self.guid = guid
        self.text = text
        self.sent_id = sent_id
        self.token_id = token_id
        self.token = token
        self.lemma = lemma
        self.pos = pos
        self.pos2 = pos2
        self.pos3 = pos3
        self.pos4 = pos4  # 있다 없다 Embedding dimention
        self.pos5 = pos5  # 있다 없다 Embedding dimention 2
        self.head = head
        self.dep = dep


# 없음, 어서, 보조용언 : 0,1,2 판별해주는 것 넣기 -> 최종 후처리에서 attention값들 0으로 만들어주기

class KlueDPInputFeatures:
    """A single set of features of data. Property names are the same names as the corresponding inputs to a model.

    Args:
        input_ids: Indices of input sequence tokens in the vocabulary.
        attention_mask: Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``: Usually ``1`` for tokens that are NOT MASKED, ``0`` for MASKED (padded)
            tokens.
        bpe_head_mask : Mask to mark the head token of bpe in aejeol
        head_ids : head ids for each aejeols on head token index
        dep_ids : dependecy relations for each aejeols on head token index
        pos_ids : pos tag for each aejeols on head token index
    """

    def __init__(
            self,
            guid: str,
            ids: List[int],
            mask: List[int],
            bpe_head_mask: List[int],
            bpe_tail_mask: List[int],
            head_ids: List[int],
            dep_ids: List[int],
            pos_ids: List[int],
            pos_ids2: List[int],
            pos_ids3: List[int],
            pos_ids4: List[int],
            pos_ids5: List[int],
    ) -> None:
        self.guid = guid
        self.input_ids = ids
        self.attention_mask = mask
        self.bpe_head_mask = bpe_head_mask
        self.bpe_tail_mask = bpe_tail_mask
        self.head_ids = head_ids
        self.dep_ids = dep_ids
        self.pos_ids = pos_ids
        self.pos_ids2 = pos_ids2
        self.pos_ids3 = pos_ids3
        self.pos_ids4 = pos_ids4
        self.pos_ids5 = pos_ids5


class KlueDPDataModule(pl.LightningDataModule):
    def __init__(self, args: argparse.Namespace, processor: DataProcessor) -> None:
        super().__init__()
        self.hparams = args
        self.processor = processor

    def prepare_dataset(self, dataset_type: str) -> Any:
        "Called to initialize data. Use the call to construct features and dataset"
        logger.info("Creating features from dataset file at %s", self.hparams.data_dir)

        if dataset_type == "train":
            dataset = self.processor.get_train_dataset(self.hparams.data_dir, self.hparams.train_file_name)
        elif dataset_type == "dev":
            dataset = self.processor.get_dev_dataset(self.hparams.data_dir, self.hparams.dev_file_name)
        elif dataset_type == "test":
            dataset = self.processor.get_test_dataset(self.hparams.data_dir, self.hparams.test_file_name)
        else:
            raise ValueError(f"{dataset_type} do not support. [train|dev|test]")
        logger.info(f"Prepare {dataset_type} dataset (Count: {len(dataset)}) ")
        return dataset

    def get_dataloader(self, dataset_type: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self.prepare_dataset(dataset_type),
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=self.processor.collate_fn,
        )

    @overrides
    def train_dataloader(self) -> DataLoader:
        return self.get_dataloader("train", self.hparams.train_batch_size, shuffle=True)

    @overrides
    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", self.hparams.eval_batch_size, shuffle=False)

    @overrides
    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", self.hparams.eval_batch_size, shuffle=False)

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser.add_argument("--data_dir", default=None, type=str, help="The input data dir", required=True)
        parser.add_argument(
            "--train_file_name",
            default=None,
            type=str,
            help="Name of the train file",
        )
        parser.add_argument(
            "--dev_file_name",
            default=None,
            type=str,
            help="Name of the dev file",
        )
        parser.add_argument(
            "--test_file_name",
            default=None,
            type=str,
            help="Name of the test file",
        )
        parser.add_argument("--num_workers", default=4, type=int, help="kwarg passed to DataLoader")
        parser.add_argument("--train_batch_size", default=32, type=int)
        parser.add_argument("--eval_batch_size", default=64, type=int)
        return parser


class KlueDPProcessor(DataProcessor):
    origin_train_file_name = "klue-dp-v1.1_train.tsv"
    origin_dev_file_name = "klue-dp-v1.1_dev.tsv"
    origin_test_file_name = "klue-dp-v1.1_test.tsv"

    datamodule_type = KlueDPDataModule

    def __init__(self, args: argparse.Namespace, tokenizer: PreTrainedTokenizer) -> None:
        super().__init__(args, tokenizer)

    def _create_examples(self, file_path: str, dataset_type: str) -> List[KlueDPInputExample]:
        sent_id = -1
        examples = []
        VV_WORD = []
        VV_POS = []
        NNG_VERB_WORD = []


        NNG_SANG_WORD = []

        VV_EC_Error_Sentences = []
        with open("VV_LIST.txt", "r", encoding="utf-8") as f1:
            for line in f1:
                parsed = line.split("\t")
                VV_WORD.append(parsed[0][:-1])
                VV_POS.append(parsed[1])
        with open("NNG_VERB_LIST_test.txt", "r", encoding="utf-8") as f2:
            for line in f2:
                NNG_VERB_WORD.append(line.replace("\n", ""))
        #with open("error_sentences2.txt","r",encoding="utf-8") as f3:
        #    for line in f3:
        #        VV_EC_Error_Sentences.append(line.replace("\n",""))
        with open("NNG_SANG_LIST_test.txt", "r", encoding="utf-8") as f3:
            for line in f3:
                NNG_SANG_WORD.append(line.replace("\n", ""))

        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line == "" or line == "\n" or line == "\t":
                    continue
                if line.startswith("#"):
                    parsed = line.strip().split("\t")
                    if len(parsed) != 2:  # metadata line about dataset
                        continue
                    else:
                        sent_id += 1
                        text = parsed[1].strip()
                        guid = parsed[0].replace("##", "").strip()
                else:
                    token_list = [token.replace("\n", "") for token in line.split("\t")] + ["-", "-"]
                    # print(token_list[6])
                    # print(token_list[3])

                    """
                    p11 = p1.search(token_list[1])
                    p22 = p2.search(token_list[1])
                    p33 = p3.search(token_list[1])
                    p44 = p4.search(token_list[1])

                    if p11 != None:
                        token_list[1] = token_list[1][:p11.span()[0]] + "을"
                    elif p22 != None:
                        token_list[1] = token_list[1][:p22.span()[0]] + "이"
                    elif p33 != None:
                        token_list[1] = token_list[1][:p33.span()[0]] + "은"
                    elif p44 != None:
                        token_list[1] = token_list[1][:p44.span()[0]] + "와"
                    """

                    """
                    if "있" in token_list[1]:
                        #example_temp = example.pos.split("+")
                        if "VA" in token_list[3]:
                            #VA_index = token_list[3].find("VA")
                            #token_list[3][VA_index:VA_index+1]="있다"
                            token_list[3] = token_list[3].replace("VA","있다")
                        elif "VX" in token_list[3]:
                            #VX_index = token_list[3].find("VX")
                            #token_list[3][VX_index:VX_index + 1] = "있다"
                            token_list[3] = token_list[3].replace("VX", "있다")
                        else:
                            token_list[3] = token_list[3].replace("VV", "있다")
                        #print(token_list[3])

                    if "없" in token_list[1]:
                        #example_temp = example.pos.split("+")
                        if "VA" in token_list[3]:
                            #VA_index = token_list[3].find("VA")
                            #token_list[3][VA_index:VA_index + 1] = "없다"
                            token_list[3] = token_list[3].replace("VA", "없다")
                        elif "VX" in token_list[3]:
                            #VX_index = token_list[3].find("VX")
                            #token_list[3][VX_index:VX_index + 1] = "없다"
                            token_list[3] = token_list[3].replace("VX", "없다")
                        else:
                            token_list[3] = token_list[3].replace("VV", "없다")
                    """

                    """
                    if token_list[3].split("+")[0] == "NNG":
                        if token_list[2].split(" ")[0] in NNG_VERB_WORD:
                            token_list.append("동작성명사")
                        elif token_list[2].split(" ")[0] in NNG_SANG_WORD:
                            token_list.append("상태성명사")
                        else:
                            token_list.append("NNG")  # NNG 3.07
                    """
                    """
                    if "NNG" in token_list[3]:
                        temp_token_list=token_list[3].split("+")
                        for i in range(len(temp_token_list)):
                            if temp_token_list[i]=="NNG":
                                if token_list[2].split(" ")[i] in NNG_VERB_WORD:
                                    temp_token_list[i] = "동작성명사"
                                    token_list.append("NULL")
                                elif token_list[2].split(" ")[i] in NNG_SANG_WORD:
                                    temp_token_list[i] = "상태성명사"
                                    token_list.append("NULL")
                                else:
                                    token_list.append("NULL")
                        token_list[3]='+'.join(temp_token_list)
                        """
                    if token_list[3].split("+")[0] == "NNG":
                        #i = NNG_VERB_WORD.index(token_list[2].split(" ")[0])
                        #tempVV = VV_POS[i].replace("\n", "")
                        # token_list.append(tempVV)
                        #token_list.append("")
                        if token_list[2].split(" ")[0] in NNG_VERB_WORD:
                            token_list.append("동작성명사")
                            token_list[3] = token_list[3].replace("NNG", "동작성명사")
                        elif token_list[2].split(" ")[0] in NNG_SANG_WORD:
                            token_list.append("상태성명사")
                            token_list[3] = token_list[3].replace("NNG", "상태성명사")
                        else:
                            token_list.append("NNG")  # NNG 3.07
                    elif token_list[3].split("+")[0] == "VV":
                        if token_list[2].split(" ")[0] in VV_WORD:
                            i = VV_WORD.index(token_list[2].split(" ")[0])
                            tempVV = VV_POS[i].replace("\n", "")
                            #token_list.append(tempVV)
                            token_list.append(tempVV)
                            token_list[3] = token_list[3].replace("VV", tempVV)
                        else:
                            token_list.append("VV")
                    else:
                        temp_token_list = token_list[3].split("+")
                        if len(temp_token_list) >= 3:
                            token_list.append(temp_token_list[0])  # VV 3.117
                        else:
                            token_list.append("0")
                    """
                    elif token_list[3].split("+")[0] == "VV":
                        if token_list[2].split(" ")[0] in VV_WORD:
                            i = VV_WORD.index(token_list[2].split(" ")[0])
                            tempVV = VV_POS[i].replace("\n", "")
                            token_list.append("NULL")
                            # token_list.append(tempVV)
                            token_list[3] = token_list[3].replace("VV", tempVV)
                        else:
                            token_list.append("NULL")  # VV 3.07
                    else:
                        token_list.append("NULL")
                    """
                    #print(len(token_list))
                    # print(token_list[8])
                    examples.append(
                        KlueDPInputExample(
                            guid=guid,
                            text=text,
                            sent_id=sent_id,
                            token_id=int(token_list[0]),
                            token=token_list[1],
                            lemma=token_list[2].replace(" ", ""),
                            pos=token_list[3],
                            pos2=token_list[3],
                            pos3=token_list[3],
                            pos4=token_list[8],
                            pos5=token_list[3],
                            head=token_list[4],
                            dep=token_list[5],
                        )
                    )

        return examples

        """
                    if text in VV_EC_Error_Sentences:
                        #print(text)
                        continue
                    else:
                        examples.append(
                            KlueDPInputExample(
                                guid=guid,
                                text=text,
                                sent_id=sent_id,
                                token_id=int(token_list[0]),
                                token=token_list[1],
                                lemma=token_list[2].replace(" ", ""),
                                pos=token_list[3],
                                pos2=token_list[3],
                                pos3=token_list[3],
                                pos4=token_list[8],
                                pos5=token_list[3],
                                head=token_list[4],
                                dep=token_list[5],
                            )
                        )

        return examples
        """

    def convert_examples_to_features(
            self,
            examples: List[KlueDPInputExample],
            tokenizer: PreTrainedTokenizer,
            max_length: int,
            pos_label_list: List[str],
            dep_label_list: List[str],
            classification_label_list: List[str],
    ) -> List[KlueDPInputFeatures]:

        pos_label_map = {label: i for i, label in enumerate(pos_label_list)}
        dep_label_map = {label: i for i, label in enumerate(dep_label_list)}
        classification_label_map = {label: i for i, label in enumerate(classification_label_list)}
        DP_POS = ["VV", "VA", "VX", "XSV", "XSA", "VCP", "JKQ", "자동사", "타동사","자타동사"]
        NNB_VX = ["것", "터", "뿐", "따름", "모양", "지경", "참", "수", "리", "만하다", "법하다", "듯하다", "걸", "말", "노릇", "예정", "길"]
        # print(pos_label_map)
        SENT_ID = 0

        token_list: List[str] = []
        pos_list: List[str] = []
        pos_list2: List[str] = []
        pos_list3: List[str] = []
        pos_list4: List[str] = []
        pos_list5: List[str] = []
        head_list: List[int] = []
        dep_list: List[str] = []

        features = []
        # print(max_length,type(max_length))
        for example in examples:
            # print(example.sent_id)
            # print(SENT_ID)
            if SENT_ID != example.sent_id:
                SENT_ID = example.sent_id
                encoded = tokenizer.encode_plus(
                    " ".join(token_list),
                    None,
                    add_special_tokens=True,
                    max_length=max_length,
                    truncation=True,
                    padding="max_length",
                )

                ids, mask = encoded["input_ids"], encoded["attention_mask"]

                bpe_head_mask = [0]
                bpe_tail_mask = [0]
                head_ids = [-1]
                dep_ids = [-1]
                pos_ids = [-1]  # --> CLS token
                pos_ids2 = [-1]
                pos_ids3 = [-1]
                pos_ids4 = [-1]
                pos_ids5 = [-1]
                # print("pos_list:", pos_list)

                for token, head, dep, pos, pos2, pos3, pos4, pos5 in zip(token_list, head_list, dep_list, pos_list,
                                                                         pos_list2,
                                                                         pos_list3, pos_list4, pos_list5):
                    bpe_len = len(tokenizer.tokenize(token))
                    head_token_mask = [1] + [0] * (bpe_len - 1)
                    tail_token_mask = [0] * (bpe_len - 1) + [1]
                    bpe_head_mask.extend(head_token_mask)
                    bpe_tail_mask.extend(tail_token_mask)

                    head_mask = [head] + [-1] * (bpe_len - 1)
                    head_ids.extend(head_mask)
                    dep_mask = [dep_label_map[dep]] + [-1] * (bpe_len - 1)
                    dep_ids.extend(dep_mask)
                    pos_mask = [pos_label_map[pos]] + [-1] * (bpe_len - 1)
                    pos_mask2 = [pos_label_map[pos2]] + [-1] * (bpe_len - 1)
                    pos_mask3 = [pos_label_map[pos3]] + [-1] * (bpe_len - 1)
                    pos_mask4 = [pos_label_map[pos4]] + [-1] * (bpe_len - 1)
                    pos_mask5 = [classification_label_map[pos5]] + [-1] * (bpe_len - 1)
                    pos_ids.extend(pos_mask)
                    pos_ids2.extend(pos_mask2)
                    pos_ids3.extend(pos_mask3)
                    pos_ids4.extend(pos_mask4)
                    pos_ids5.extend(pos_mask5)
                    # print("pos_ids:",pos_ids)
                    # print("pos_ids2:", pos_ids2)

                bpe_head_mask.append(0)
                bpe_tail_mask.append(0)
                head_ids.append(-1)
                dep_ids.append(-1)
                pos_ids.append(-1)  # END token
                pos_ids2.append(-1)  # END token
                pos_ids3.append(-1)  # END token
                pos_ids4.append(-1)  # END token
                pos_ids5.append(-1)  # END token
                if len(bpe_head_mask) > max_length:
                    bpe_head_mask = bpe_head_mask[:max_length]
                    bpe_tail_mask = bpe_tail_mask[:max_length]
                    head_ids = head_ids[:max_length]
                    dep_ids = dep_ids[:max_length]
                    pos_ids = pos_ids[:max_length]
                    pos_ids2 = pos_ids2[:max_length]
                    pos_ids3 = pos_ids3[:max_length]
                    pos_ids4 = pos_ids4[:max_length]
                    pos_ids5 = pos_ids5[:max_length]
                else:
                    bpe_head_mask.extend([0] * (max_length - len(bpe_head_mask)))  # padding by max_len
                    bpe_tail_mask.extend([0] * (max_length - len(bpe_tail_mask)))  # padding by max_len
                    head_ids.extend([-1] * (max_length - len(head_ids)))  # padding by max_len
                    dep_ids.extend([-1] * (max_length - len(dep_ids)))  # padding by max_len
                    pos_ids.extend([-1] * (max_length - len(pos_ids)))
                    pos_ids2.extend([-1] * (max_length - len(pos_ids2)))
                    pos_ids3.extend([-1] * (max_length - len(pos_ids3)))
                    pos_ids4.extend([-1] * (max_length - len(pos_ids4)))
                    pos_ids5.extend([-1] * (max_length - len(pos_ids5)))

                feature = KlueDPInputFeatures(
                    guid=example.guid,
                    ids=ids,
                    mask=mask,
                    bpe_head_mask=bpe_head_mask,
                    bpe_tail_mask=bpe_tail_mask,
                    head_ids=head_ids,
                    dep_ids=dep_ids,
                    pos_ids=pos_ids,
                    pos_ids2=pos_ids2,
                    pos_ids3=pos_ids3,
                    pos_ids4=pos_ids4,
                    pos_ids5=pos_ids5,
                )
                features.append(feature)

                token_list = []
                pos_list = []
                pos_list2 = []
                pos_list3 = []
                pos_list4 = []
                pos_list5 = []
                head_list = []
                dep_list = []

            token_list.append(example.token)
            """
            if p5.search(example.token) != None:  # 에게 는 : [-1] -> [-2] 참조
                pos_list.append(example.pos.split("+")[-1])  # 맨 뒤 바로앞 pos정보 사용
                pos_list2.append(example.pos2.split("+")[0])
                pos_list3.append("0")  # Null
            else:
            """
            pos_list.append(example.pos.split("+")[-1])  # 맨 뒤 pos정보 사용
            if len(example.pos2.split("+")) > 2:
                pos_list2.append(example.pos2.split("+")[-2])  # 맨 뒤 바로앞 pos정보 사용
                pos_list3.append(example.pos3.split("+")[0])  # 맨 앞 pos정보 사용
            elif len(example.pos2.split("+")) == 2:
                    #print(example.pos2)
                pos_list2.append(example.pos2.split("+")[0])  # 맨 앞 pos정보 사용
                pos_list3.append("0")  # Null
            elif len(example.pos2.split("+")) == 1:
                pos_list2.append("0")  # Null
                pos_list3.append("0")  # Null
                # print(pos_list3)
            """
            if p6.search(example.token) != None:  # 3.11 나중에 ~고 찾아서 넣는 것으로 수정해야
                pos_list5.append("고")
                # pos_list5.append("0")           #3.13 이거 없에야하는 line, 어서 결과만 보고 나중에 삭제해야
                # print(example.token)
            else:
                pos_list5.append("0")
            """
            VP_check=example.pos4.split("+")[0]
            #print(VP_check)
            temp_pos = example.pos.split("+")
            temp_pos2 = []
            for i in range(len(temp_pos)):
                if temp_pos[i] in DP_POS:
                    temp_pos2.append(temp_pos[i])

            temp_pos3 = []

            for i in range(len(NNB_VX)):
                if NNB_VX[i] in example.token:
                    temp_pos3.append(NNB_VX[i])
            Inyong_start=[]
            sentence_list = example.text.split(" ")
            if "\"" in example.token and (example.token[-1] == "고" or example.token[-1] == "며" or example.token[-1] == "서"):  # or word_list[int(index)-1][-1]=="라") :
                if len(Inyong_start) == 0:
                    for i in range(len(sentence_list)):
                        if ("\"") in sentence_list[i] and (example.token[-1] == "고" or example.token[-1] == "며" or example.token[-1] == "서"):
                            Inyong_start.append(i)

            #sentence_list = example.text.split(" ")
            HassDa = re.compile("했다.")
            # if temp_pos[0]=="동작성명사" or temp_pos[0]=="상태성명사":
            # pos_list5.append("NOT-VP")
            if ("VV" in temp_pos) or ("VA" in temp_pos) or ("VX" in temp_pos):
                if ("있다" in example.token) or ("없다" in example.token):  # or ("있다" in example.token)
                    pos_list5.append("VV-VA-VX-있없다")
                elif HassDa.match(sentence_list[-1]) and ("VV" in temp_pos):
                    pos_list5.append("VV-VX-했다")
                elif ("했다" in example.token) and ("VX" in temp_pos):
                    pos_list5.append("VV-VX-했다")
                else:
                    pos_list5.append("0")
            elif (temp_pos[0]=="NNB" and len(temp_pos3)!=0) and not "”" in example.token and not "\"" in example.token:
                pos_list5.append("NNB-VX") #나중에 중이다 추가해보기 3-28
            elif (temp_pos[0] == "NNB" and example.token[0] == "중") and not "”" in example.token and not "\"" in example.token:
                pos_list5.append("NNB-VX(중)")
                #print(example.token)
            #elif "\"" in example.token and not int(ids[j]) >= (Inyong_start[0] + 1) :#and not int(ids[j]) + 1 == int(index): #입력어절이 인용어절보다 앞에있을경우, 두번째 "일경우
            elif "\"" in example.token:
                #    pos_list5.append("인용시작")
                if example.token[-1] == "고" or  example.token[-1] == "서" or example.token[-1] == "도":
                    pos_list5.append("인용끝")
                    #print(example.token)
                elif example.token[-1] == "며" and "다" not in example.token:
                    pos_list5.append("인용끝")
                else:
                    pos_list5.append("인용시작")
                    #print(example.token)
                #print(example.token)
            else:
                pos_list5.append("0")

            #if "인용시작" in pos_list5:
            #    print(pos_list5)
            """
            elif ("VCN" in temp_pos) or ("VCP" in temp_pos):
                if ("VCN" in temp_pos) and ("아니다" in example.token):
                    pos_list5.append("VCP-VCN-이아니다다")
                elif ("NNB+VCP" in temp_pos) and ("이다" in example.token):
                    pos_list5.append("VCP-VCN-이아니다다")
                else:
                    pos_list5.append("0")
            
            else:
                pos_list5.append("0")
            #elif ("VV" in temp_pos)
            """



            #print(VP_check)
            c = 0
            """
            if len(temp_pos2) == 0:  # VP에 관련된 형태소가 없는 경우
                pos_list5.append("NOT-VP")  # NNG도 없고 temp_pos2(동사 형태소 넣은 리스트) 길이도 0일때 -> Not-vp
                # print(example.pos)
            else:  # temp_pos2(동사 형태소 넣은 리스트) 길이가 0이 아닐 때 -> vp
                # print(2)
                # print(example.pos)
                pos_list5.append("0")  # VP
                c = c + 1
                """ #3.24
            """
                if temp_pos[0] == "NNG":
                    if VP_check == "동작성명사" or VP_check == "상태성명사":
                        #print(VP_check)
                        pos_list5.append("0") #첫번째 pos가 NNG이고 동작성 or 상태성 명사일때 -> VP
                        c = c + 1
                    else:
                        pos_list5.append("NOT-VP")  #첫번째 pos가 NNG이지만 동작성 or 상태성 명사가 아닐때 -> Not-vp
                        c = c + 1
                else:
                    pos_list5.append("NOT-VP") #NNG도 없고 temp_pos2(동사 형태소 넣은 리스트) 길이도 0일때 -> Not-vp
                    c = c + 1
                    
            else: #temp_pos2(동사 형태소 넣은 리스트) 길이가 0이 아닐 때 -> vp
                #print(2)
                # print(example.pos)
                pos_list5.append("0") #VP
                c = c + 1
                """
            #print(pos_list5)

            # print(len(pos_list5))
            #print(c)



            example_pos_temp = example.pos.split("+")
            #if example.pos4 != "NULL":
            #    pos_list4.append(example.pos4)
            #    # print(example.pos4)

            if "있" in example.token:
                if "VA" in example_pos_temp:
                    pos_list4.append("있다")
                elif "VX" in example_pos_temp:
                    pos_list4.append("있다")
                else:
                    pos_list4.append("있다")
            elif "없" in example.token:
                if "VA" in example_pos_temp:
                    pos_list4.append("없다")
                elif "VX" in example_pos_temp:
                    pos_list4.append("없다")
                else:
                    pos_list4.append("없다")
            elif "어서" in example.lemma or "아서" in example.lemma:
                pos_list4.append("어서")
            else:
                pos_list4.append("0")

            """
            elif "않" in example.token:
                pos_list4.append("않다")

            elif "어서" in example.lemma or "아서" in example.lemma:
                pos_list4.append("어서")
            else:
                pos_list4.append("0")
                """

            head_list.append(int(example.head))
            dep_list.append(example.dep)
            # print("pos_list:", pos_list)
        # print("pos_list:",pos_list)
        # print("pos_list2:", pos_list2)
        # print(" ".join(token_list))
        encoded = tokenizer.encode_plus(
            " ".join(token_list),
            None,
            add_special_tokens=True,
            max_length=int(max_length),
            truncation=True,
            padding="max_length",
        )

        ids, mask = encoded["input_ids"], encoded["attention_mask"]

        bpe_head_mask = [0]
        bpe_tail_mask = [0]
        head_ids = [-1]
        dep_ids = [-1]
        pos_ids = [-1]  # --> CLS token
        pos_ids2 = [-1]
        pos_ids3 = [-1]
        pos_ids4 = [-1]
        pos_ids5 = [-1]

        for token, head, dep, pos, pos2, pos3, pos4, pos5 in zip(token_list, head_list, dep_list, pos_list, pos_list2,
                                                                 pos_list3, pos_list4, pos_list5):
            bpe_len = len(tokenizer.tokenize(token))
            head_token_mask = [1] + [0] * (bpe_len - 1)
            tail_token_mask = [0] * (bpe_len - 1) + [1]
            bpe_head_mask.extend(head_token_mask)
            bpe_tail_mask.extend(tail_token_mask)

            head_mask = [head] + [-1] * (bpe_len - 1)
            head_ids.extend(head_mask)
            dep_mask = [dep_label_map[dep]] + [-1] * (bpe_len - 1)
            dep_ids.extend(dep_mask)
            pos_mask = [pos_label_map[pos]] + [-1] * (bpe_len - 1)
            pos_mask2 = [pos_label_map[pos2]] + [-1] * (bpe_len - 1)
            pos_mask3 = [pos_label_map[pos3]] + [-1] * (bpe_len - 1)
            pos_mask4 = [pos_label_map[pos4]] + [-1] * (bpe_len - 1)
            pos_mask5 = [classification_label_map[pos5]] + [-1] * (bpe_len - 1)

            # print(pos)

            pos_ids.extend(pos_mask)
            pos_ids2.extend(pos_mask2)
            pos_ids3.extend(pos_mask3)
            pos_ids4.extend(pos_mask4)
            pos_ids5.extend(pos_mask5)
            # print("pos_ids:", pos_ids)
            # print("pos_ids2:", pos_ids2)

        bpe_head_mask.append(0)
        bpe_tail_mask.append(0)
        head_ids.append(-1)
        dep_ids.append(-1)  # END token
        bpe_head_mask.extend([0] * (max_length - len(bpe_head_mask)))  # padding by max_len
        bpe_tail_mask.extend([0] * (max_length - len(bpe_tail_mask)))  # padding by max_len
        head_ids.extend([-1] * (max_length - len(head_ids)))  # padding by max_len
        dep_ids.extend([-1] * (max_length - len(dep_ids)))  # padding by max_len
        pos_ids.extend([-1] * (max_length - len(pos_ids)))
        pos_ids2.extend([-1] * (max_length - len(pos_ids2)))
        pos_ids3.extend([-1] * (max_length - len(pos_ids3)))
        pos_ids4.extend([-1] * (max_length - len(pos_ids4)))
        pos_ids5.extend([-1] * (max_length - len(pos_ids5)))
        # print(pos_ids)
        # print(pos_ids2)
        feature = KlueDPInputFeatures(
            guid=example.guid,
            ids=ids,
            mask=mask,
            bpe_head_mask=bpe_head_mask,
            bpe_tail_mask=bpe_tail_mask,
            head_ids=head_ids,
            dep_ids=dep_ids,
            pos_ids=pos_ids,
            pos_ids2=pos_ids2,
            pos_ids3=pos_ids3,
            pos_ids4=pos_ids4,
            pos_ids5=pos_ids5,
        )
        features.append(feature)
        return features

    def _convert_features(self, examples: List[KlueDPInputExample]) -> List[KlueDPInputFeatures]:
        return self.convert_examples_to_features(
            examples,
            self.tokenizer,
            max_length=max_seq_length2,
            dep_label_list=get_dep_labels(),
            pos_label_list=get_pos_labels(),
            classification_label_list=get_classification_labels()
        )

    # self.hparams.max_seq_length,
    def _create_dataset(self, file_path: str, dataset_type: str) -> TensorDataset:
        examples = self._create_examples(file_path, dataset_type)
        features = self._convert_features(examples)

        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
        all_bpe_head_mask = torch.tensor([f.bpe_head_mask for f in features], dtype=torch.long)
        all_bpe_tail_mask = torch.tensor([f.bpe_tail_mask for f in features], dtype=torch.long)
        all_head_ids = torch.tensor([f.head_ids for f in features], dtype=torch.long)
        all_dep_ids = torch.tensor([f.dep_ids for f in features], dtype=torch.long)
        all_pos_ids = torch.tensor([f.pos_ids for f in features], dtype=torch.long)
        all_pos_ids2 = torch.tensor([f.pos_ids2 for f in features], dtype=torch.long)
        all_pos_ids3 = torch.tensor([f.pos_ids3 for f in features], dtype=torch.long)
        all_pos_ids4 = torch.tensor([f.pos_ids4 for f in features], dtype=torch.long)
        all_pos_ids5 = torch.tensor([f.pos_ids5 for f in features], dtype=torch.long)

        return TensorDataset(
            all_input_ids,
            all_attention_mask,
            all_bpe_head_mask,
            all_bpe_tail_mask,
            all_head_ids,
            all_dep_ids,
            all_pos_ids,
            all_pos_ids2,
            all_pos_ids3,
            all_pos_ids4,
            all_pos_ids5,
        )

    @overrides
    def get_train_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_train_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "train")

    @overrides
    def get_dev_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "dev")

    @overrides
    def get_test_dataset(self, data_dir: str, file_name: Optional[str] = None) -> TensorDataset:
        file_path = os.path.join(data_dir, file_name or self.origin_test_file_name)

        if not os.path.exists(file_path):
            logger.info("Test dataset doesn't exists. So loading dev dataset instead.")
            file_path = os.path.join(data_dir, self.hparams.dev_file_name or self.origin_dev_file_name)

        logger.info(f"Loading from {file_path}")
        return self._create_dataset(file_path, "test")

    @overrides
    def get_labels(self) -> List[str]:
        return get_dep_labels()

    def collate_fn(self, batch: List[Tuple]) -> Tuple[torch.Tensor, Any, Any, Any]:
        # 1. set args
        batch_size = len(batch)
        pos_padding_idx = None if self.hparams.no_pos else len(get_pos_labels())
        pos_padding_idx2 = None if self.hparams.no_pos else len(get_pos_labels())
        pos_padding_idx3 = None if self.hparams.no_pos else len(get_pos_labels())
        pos_padding_idx4 = None if self.hparams.no_pos else len(get_pos_labels())
        pos_padding_idx5 = None if self.hparams.no_pos else len(get_classification_labels())
        # 2. build inputs : input_ids, attention_mask, bpe_head_mask, bpe_tail_mask
        batch_input_ids = []
        batch_attention_masks = []
        batch_bpe_head_masks = []
        batch_bpe_tail_masks = []
        for batch_id in range(batch_size):
            (
                input_id,
                attention_mask,
                bpe_head_mask,
                bpe_tail_mask,
                _,
                _,
                _,
                _,
                _,
                _,
                _,
            ) = batch[batch_id]
            batch_input_ids.append(input_id)
            batch_attention_masks.append(attention_mask)
            batch_bpe_head_masks.append(bpe_head_mask)
            batch_bpe_tail_masks.append(bpe_tail_mask)
        # 2. build inputs : packing tensors
        # 나는 밥을 먹는다. => [CLS] 나 ##는 밥 ##을 먹 ##는 ##다 . [SEP]
        # input_id : [2, 717, 2259, 1127, 2069, 1059, 2259, 2062, 18, 3, 0, 0, ...]
        # bpe_head_mask : [0, 1, 0, 1, 0, 1, 0, 0, 0, 0, ...] (indicate word start (head) idx)
        input_ids = torch.stack(batch_input_ids)
        attention_masks = torch.stack(batch_attention_masks)
        bpe_head_masks = torch.stack(batch_bpe_head_masks)
        bpe_tail_masks = torch.stack(batch_bpe_tail_masks)
        # 3. token_to_words : set in-batch max_word_length
        max_word_length = max(torch.sum(bpe_head_masks, dim=1)).item()
        # 3. token_to_words : placeholders
        head_ids = torch.zeros(batch_size, max_word_length).long()
        type_ids = torch.zeros(batch_size, max_word_length).long()
        pos_ids = torch.zeros(batch_size, max_word_length + 1).long()
        pos_ids2 = torch.zeros(batch_size, max_word_length + 1).long()
        pos_ids3 = torch.zeros(batch_size, max_word_length + 1).long()
        pos_ids4 = torch.zeros(batch_size, max_word_length + 1).long()
        pos_ids5 = torch.zeros(batch_size, max_word_length + 1).long()
        mask_e = torch.zeros(batch_size, max_word_length + 1).long()
        # 3. token_to_words : head_ids, type_ids, pos_ids, mask_e, mask_d
        for batch_id in range(batch_size):
            (
                _,
                _,
                bpe_head_mask,
                _,
                token_head_ids,
                token_type_ids,
                token_pos_ids,
                token_pos_ids2,
                token_pos_ids3,
                token_pos_ids4,
                token_pos_ids5,
            ) = batch[batch_id]
            # head_id : [1, 3, 5] (prediction candidates)
            # token_head_ids : [-1, 3, -1, 3, -1, 0, -1, -1, -1, .-1, ...] (ground truth head ids)
            head_id = [i for i, token in enumerate(bpe_head_mask) if token == 1]
            word_length = len(head_id)
            head_id.extend([0] * (max_word_length - word_length))
            head_ids[batch_id] = token_head_ids[head_id]
            type_ids[batch_id] = token_type_ids[head_id]
            if not self.hparams.no_pos:
                pos_ids[batch_id][0] = torch.tensor(pos_padding_idx)
                pos_ids[batch_id][1:] = token_pos_ids[head_id]
                pos_ids[batch_id][int(torch.sum(bpe_head_mask)) + 1:] = torch.tensor(pos_padding_idx)

                pos_ids2[batch_id][0] = torch.tensor(pos_padding_idx2)
                pos_ids2[batch_id][1:] = token_pos_ids2[head_id]
                pos_ids2[batch_id][int(torch.sum(bpe_head_mask)) + 1:] = torch.tensor(pos_padding_idx2)

                pos_ids3[batch_id][0] = torch.tensor(pos_padding_idx3)
                pos_ids3[batch_id][1:] = token_pos_ids3[head_id]
                pos_ids3[batch_id][int(torch.sum(bpe_head_mask)) + 1:] = torch.tensor(pos_padding_idx3)

                pos_ids4[batch_id][0] = torch.tensor(pos_padding_idx4)
                pos_ids4[batch_id][1:] = token_pos_ids4[head_id]
                pos_ids4[batch_id][int(torch.sum(bpe_head_mask)) + 1:] = torch.tensor(pos_padding_idx4)

                pos_ids5[batch_id][0] = torch.tensor(pos_padding_idx5)
                pos_ids5[batch_id][1:] = token_pos_ids5[head_id]
                pos_ids5[batch_id][int(torch.sum(bpe_head_mask)) + 1:] = torch.tensor(pos_padding_idx5)
            mask_e[batch_id] = torch.LongTensor([1] * (word_length + 1) + [0] * (max_word_length - word_length))
        mask_d = mask_e[:, 1:]
        # 4. pack everything
        masks = (attention_masks, bpe_head_masks, bpe_tail_masks, mask_e, mask_d)
        ids = (head_ids, type_ids, pos_ids, pos_ids2, pos_ids3, pos_ids4, pos_ids5)

        return input_ids, masks, ids, max_word_length

    @staticmethod
    def add_specific_args(parser: argparse.ArgumentParser, root_dir: str) -> argparse.ArgumentParser:
        parser = KlueDataModule.add_specific_args(parser, root_dir)
        parser.add_argument(
            "--max_seq_length",
            default=128,
            type=int,
            help="The maximum total input sequence length after tokenization. Sequences longer "
                 "than this will be truncated, sequences shorter will be padded.",
        )
        return parser


def get_dep_labels() -> List[str]:
    """
    label for dependency relations format:

    {structure}_(optional){function}

    """
    dep_labels = [
        "NP",
        "NP_AJT",
        "VP",
        "NP_SBJ",
        "VP_MOD",
        "NP_OBJ",
        "AP",
        "NP_CNJ",
        "NP_MOD",
        "VNP",
        "DP",
        "DP_AJT",  # DP_AJT~DP_CMP 1.22 중민 추가
        "DP_MOD",
        "DP_SBJ",
        "DP_CMP",
        "VP_AJT",
        "VNP_MOD",
        "NP_CMP",
        "VP_SBJ",
        "VP_CMP",
        "VP_OBJ",
        "VNP_CMP",
        "AP_MOD",
        "X_AJT",
        "VP_CNJ",
        "VNP_AJT",
        "IP",
        "IP_CMP",  # IP_CMP~IP_OBJ 1.22 중민 추가
        "IP_AJT",
        "IP_SBJ",
        "IP_CNJ",
        "IP_MOD",
        "IP_OBJ",
        "X",
        "X_SBJ",
        "VNP_OBJ",
        "VNP_SBJ",
        "X_OBJ",
        "AP_AJT",
        "L",
        "X_MOD",
        "X_CNJ",
        "VNP_CNJ",
        "X_CMP",
        "AP_CMP",
        "AP_SBJ",
        "R",
        "NP_SVJ",
        "AP_OBJ",  # 22.01.11 중민 추가
        "AP_CNJ",  # 22.01.11 중민 추가
        "L_MOD",  # 이부분 나중에 데이터셋에서 없에든지 해야
    ]
    return dep_labels


def get_pos_labels() -> List[str]:
    """label for part-of-speech tags"""
    # "이다", #VCP로 됨
    # "아니다", #VCN으로 됨
    return [
        "0",
        "있다",
        "없다",
        "어서",
        "자동사",
        "타동사",
        "자타동사",
        "동작성명사",
        "상태성명사",
        "NNG",
        "NNP",
        "NNB",
        "NP",
        "NR",
        "VV",
        "VA",
        "VX",
        "VCP",
        "VCN",
        "MMA",
        "MMD",
        "MMN",
        "MAG",
        "MAJ",
        "JC",
        "IC",
        "JKS",
        "JKC",
        "JKG",
        "JKO",
        "JKB",
        "JKV",
        "JKQ",
        "JX",
        "EP",
        "EF",
        "EC",
        "ETN",
        "ETM",
        "XPN",
        "XSN",
        "XSV",
        "XSA",
        "XR",
        "SF",
        "SP",
        "SS",
        "SE",
        "SO",
        "SL",
        "SH",
        "SW",
        "SN",
        "NA",
        "NF",  # 중민 추가
        "NV",
        # "고",
    ]


def get_classification_labels() -> List[str]:
    return [
        "0",
        "NOT-VP",
        "VV-VA-VX-있없다",
        "VCP-VCN-이아니다다",
        "VV-VX-했다",
        "NNB-VX",
        "인용시작",
        "인용끝",
        "NNB-VX(중)",
        #"VP",
   ]