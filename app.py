import streamlit as st
import streamlit.components.v1 as components
import torch

from transformers import MBart50TokenizerFast, MBartForConditionalGeneration

from pyvis.network import Network

#from constants import * 

lang_codes_by_name = \
{'Arabic': 'ar_AR',
 'Czech': 'cs_CZ',
 'German': 'de_DE',
 'English': 'en_XX',
 'Spanish': 'es_XX',
 'Estonian': 'et_EE',
 'Finnish': 'fi_FI',
 'French': 'fr_XX',
 'Gujarati': 'gu_IN',
 'Hindi': 'hi_IN',
 'Italian': 'it_IT',
 'Japanese': 'ja_XX',
 'Kazakh': 'kk_KZ',
 'Korean': 'ko_KR',
 'Lithuanian': 'lt_LT',
 'Latvian': 'lv_LV',
 'Burmese': 'my_MM',
 'Nepali': 'ne_NP',
 'Dutch': 'nl_XX',
 'Romanian': 'ro_RO',
 'Russian': 'ru_RU',
 'Sinhala': 'si_LK',
 'Turkish': 'tr_TR',
 'Vietnamese': 'vi_VN',
 'Chinese': 'zh_CN',
 'Afrikaans': 'af_ZA',
 'Azerbaijani': 'az_AZ',
 'Bengali': 'bn_IN',
 'Persian': 'fa_IR',
 'Hebrew': 'he_IL',
 'Croatian': 'hr_HR',
 'Indonesian': 'id_ID',
 'Georgian': 'ka_GE',
 'Khmer': 'km_KH',
 'Macedonian': 'mk_MK',
 'Malayalam': 'ml_IN',
 'Mongolian': 'mn_MN',
 'Marathi': 'mr_IN',
 'Polish': 'pl_PL',
 'Pashto': 'ps_AF',
 'Portuguese': 'pt_XX',
 'Swedish': 'sv_SE',
 'Swahili': 'sw_KE',
 'Tamil': 'ta_IN',
 'Telugu': 'te_IN',
 'Thai': 'th_TH',
 'Tagalog': 'tl_XX',
 'Ukrainian': 'uk_UA',
 'Urdu': 'ur_PK',
 'Xhosa': 'xh_ZA',
 'Galician': 'gl_ES',
 'Slovene': 'sl_SI'}

@st.cache(allow_output_mutation=True)
def load_model():
    return MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

@st.cache(allow_output_mutation=True)
def load_tokenizer():
    return MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt", use_fast=False)

class Translation():
    def __init__(self, src_lang, dest_lang):
        self.model = load_model()
        self.tokenizer = load_tokenizer()
        self.tokenizer.src_lang = src_lang
        self.dest_lang = dest_lang

    def process(self, src_text):
        encoded = self.tokenizer(src_text, return_tensors="pt")
        generated_tokens = self.model.generate(**encoded, forced_bos_token_id=self.tokenizer.lang_code_to_id[self.dest_lang])
        generated_texts = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        self.dest_text = generated_texts[0]

        encoder_input_ids = self.tokenizer(src_text, return_tensors="pt").input_ids
        decoder_input_ids = self.tokenizer(self.dest_text, return_tensors="pt").input_ids

        self.outputs = self.model(input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids, output_attentions=True)

        self.encoder_text = self.tokenizer.convert_ids_to_tokens(encoder_input_ids[0])
        self.decoder_text = self.tokenizer.convert_ids_to_tokens(decoder_input_ids[0])

        mean_cross_attentions = tuple([torch.mean(self.outputs.cross_attentions[0], 1, True)])
        self.positions = [int(i) for i in torch.argmax(mean_cross_attentions[0], dim=2).flatten()] # 語順の並び変えの配列
        
class TranslationNetwork():
    def __init__(self, network):
        self.network = network
        self.n_nodes = 0
        self.n_src_nodes = 0
        self.n_dest_nodes = 0
        
    def add_nodes(self, words, group):
        if group == "src":
            self.n_src_nodes = len(words)
            group_i = 0
            hidden_nodes_i = [0, self.n_src_nodes-1]
        elif group == "dest":
            self.n_dest_nodes = len(words)
            group_i = 1
            hidden_nodes_i = [0, self.n_dest_nodes-1]
        self.hidden_edges_i = [0, self.n_src_nodes-1, self.n_src_nodes, self.n_src_nodes+self.n_dest_nodes-1]

        size = 10
        x_margin, y_margin = 100, 100        
        for i, word in enumerate(words):
            hidden = i in hidden_nodes_i
            self.network.add_node(self.n_nodes, shape="square", label=word, group=f"{group}", x=i*x_margin, y=group_i*y_margin, size=size, physics=False, hidden=hidden)
            self.n_nodes += 1

    def add_edges(self, positions):
        for i, position in enumerate(positions):
            j = self.n_src_nodes + position
            hidden = i in self.hidden_edges_i or j in self.hidden_edges_i
            self.network.add_edge(i, j, color="gray", hidden=hidden)        

st.set_page_config(layout="wide")
st.title("The Word Order Comparison of Translation")

src_lang_name = st.selectbox("Source Language", list(lang_codes_by_name.keys()), index=3)
tgt_lang_name = st.selectbox("Target Language", list(lang_codes_by_name.keys()), index=11)
            
with st.spinner("Loading the model"):
    src_lang, tgt_lang = lang_codes_by_name[src_lang_name], lang_codes_by_name[tgt_lang_name]
    translation = Translation(src_lang, tgt_lang)

src_text = st.text_input("Original Text", "I saw a girl with a telescope in the garden.")

if st.button("Translate"):
    with st.spinner("Translating..."):
        translation.process(src_text)

        st.subheader("Translated")
        st.write(translation.dest_text)

        tn = TranslationNetwork(Network(width="100%", height="300px"))
        tn.add_nodes(translation.encoder_text, group="src")
        tn.add_nodes(translation.decoder_text, group="dest")
        tn.add_edges(translation.positions)   
        
        fname = f"{src_text}_{src_lang_name}_{tgt_lang_name}.html"
        tn.network.show(fname)
        html_file = open(fname, "r", encoding="utf-8")
        components.html(html_file.read(), height=500)

