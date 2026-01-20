import spacy 
# pip install spacy -
# spacy.cli.download("en_core_web_sm") # It is required only once to run

# This is an english language model that contains, vocabs, grammar rules, word vectors
nlp = spacy.load("en_core_web_sm")

# text = "The studies were running better than expected"

text = """
Based on coalescence of Mitochondrial DNA and Y Chromosome data, it is thought that the earliest extant lineages of anatomically modern humans or Homo sapiens on the Indian subcontinent had reached there from Africa between 80,000 and 50,000 years ago, and with high likelihood by 55,000 years ago.[26][27][28][81] Their long occupation, initially in varying forms of isolation as hunter-gatherers, has made the region highly diverse, second only to Africa in human genetic diversity.[29] However, the earliest known modern human fossils in South Asia date to about 30,000 years ago.[27] Evidence for the neolithic period appeared in the western margins of the Indus river basin, in Mehrgarh, Balochistan, Pakistan after 7000 BCE. Domestication of grain-producing plants (including barley) and animals (including humped zebu cattle) occurred here. These cultures gradually evolved into the Indus Valley Civilisation, which flourished during 2500–1900 BCE in Pakistan and western India.[82][30] Centred around cities such as Mohenjo-daro, Harappa, Dholavira, Ganweriwala, and Rakhigarhi,[83] its characteristic features included standardised weights; steatite seals; a written script; arts and crafts including pottery styles, terracotta human and animal statuettes; urban planning; and public works.[83] Networks of towns and villages grew around the cities in a new agro-pastoral economy."""

doc = nlp(text)
print(doc)

for token in doc:
    print(f'{token.text}, ----> {token.lemma_}')