import spacy
from sklearn.feature_extraction.text import CountVectorizer # Bag of Words
from sklearn.decomposition import LatentDirichletAllocation # Topic Modelling
from collections import Counter
from warnings import filterwarnings
from transformers import pipeline

filterwarnings('ignore')

class FeedbackAnalyzer:
    def __init__(self):
        print(f'Loding Models....(This step can take some time) ')
        self.sentiment_analyzer = pipeline('sentiment-analysis')
        # self.nlp = spacy.load('en_core_web_sm')
        self.nlp = spacy.load('en_core_web_lg')
        print(f'Models Loaded Successfully')

    def analyze_sentiment(self, reviews):
        results = []
        for review in reviews:
            sentiment = self.sentiment_analyzer(review)[0]
            results.append({
                'review': review,
                'label': sentiment['label'],
                'confidence': round(sentiment['score'],2)
            })
        return results
    
    def extract_entities(self, reviews):
        # We prepared a bucket to store entities 
        all_entities = {
            'PRODUCT': [],
            'ORG': [],
            'GPE': [],
            'PERSON': []
        }

        for review in reviews:
            doc = self.nlp(review)
            for ent in doc.ents:
                if ent.label_ in all_entities: # We will store only the required entity types
                    all_entities[ent.label_].append(ent.text) # save entity text

        entity_counts = {}
        for ent_type, entities in all_entities.items():
            if entities:
                entity_counts[ent_type] = Counter(entities).most_common(5)

        return entity_counts
    
    def discover_topics(self, reviews, num_topic = 3):
        vectorizer = CountVectorizer(stop_words='english')
        try:
            doc_matrix = vectorizer.fit_transform(reviews) # bag of word matrix
            lda = LatentDirichletAllocation(n_components=num_topic, random_state = 42, max_iter=20)
            lda.fit(doc_matrix)
            words = vectorizer.get_feature_names_out()
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_word_idx = topic.argsort()[-5:][::-1]
                top_words = [words[i] for i in top_word_idx]
                topics.append({
                    'topic_number': topic_idx + 1,
                    'keywords': top_words
                })
            return topics
        except Exception as e:
            print(f'Error in topic Discovery: {e}')

    def get_summary_stats(self, sentiment_results):
        sentiments = [r['label'] for r in sentiment_results]
        total = len(sentiments)
        positive = sentiments.count('POSITIVE')
        negative = sentiments.count('NEGATIVE')
        return {
            'total_reviews': total,
            'positive_reviews': positive,
            'negative_reviews': negative,
            'positive_percentange': round(positive/total * 100, 2),
            'negative_percentage': round(negative/total * 100, 2)
        }
    
    def analyze_all(self,reviews): 
        print("=="*50)
        print(f'CUSTOMER FEEDBACK ANALYSIS REPORT')
        print("=="*50)

        # 1. Sentiment Analysis
        print(f'1. Sentiment Analysis')
        sentiment_results = self.analyze_sentiment(reviews)
        stats = self.get_summary_stats(sentiment_results)

        # 2. Entity Extraction
        print(f'2. Entity Extraction')
        entities = self.extract_entities(reviews)

        # 3. Topic Discovery -- Topic Modelling
        print(f'3. Topic Discovery')
        topics = self.discover_topics(reviews)

        print(f'Analysis Complete!')

        return {
            'sentiment_results': sentiment_results,
            'stats': stats,
            'entities': entities,
            'topics': topics
        }
    

# Visulaziation of our result
def print_results(results):
    print(f'=='*50)
    print('Summary Statitics:')
    stats = results['stats']
    print(f'Total Reviews: {stats['total_reviews']}')
    print(f'Positive Reviews: {stats['positive_reviews']} ({stats['positive_percentange']}%)')
    print(f'Negative Reviews: {stats['negative_reviews']} ({stats['negative_percentage']}%)')

    # Sentiment Details
    print("\n" + "=="*20 + "Individual Review Sentiment" + "=="*20)
    for i, result in enumerate(results['sentiment_results'][:5],1):
        sentiment_emoji = "😀" if result['label'] == 'POSITIVE' else "😒"
        print(f'\n{i}. Review: {result['review']}\n   Sentiment: {result['label']} {sentiment_emoji} (Confidence: {result['confidence']})')
    if len(results['sentiment_results']) > 5:
        print(f'\n ... and {len(results['sentiment_results']) - 5} more reviews analyzed')

    # Topic Details
    print("\n" + "=="*20 + "Topic Discovery" + "=="*20)
    for topic in results['topics']:
        print(f'Topic {topic['topic_number']}: ' + ','.join(topic['keywords']))

    # Entities Details
    print("\n" + "=="*20 + "Extracted Entities" + "=="*20)
    entity_labels = {
        "PRODUCT": 'Products',
        "ORG": 'Organization',
        "GPE": 'Locations',
        "PERSON": "People Mentioned"
    }
    entities = results['entities']
    if entities:
        for ent_type, labels in entity_labels.items():
            if ent_type in entities:
                print(f'\n{labels}')
                for entity, count in entities[ent_type]:
                    print(f" - {entity} (mentioned {count} times)")
    else:
        print("No Significant entities found")

if __name__ == '__main__':

    sample_reviews = [
    "Amazon customer support was excellent. Rahul from Bangalore fixed my Echo Dot issue quickly.",
    "The Samsung Galaxy S24 Ultra has great performance and an outstanding camera.",
    "Flipkart delivered my Lenovo ThinkPad to Mumbai on time and in good condition.",
    "Reliance Digital customer service was poor. Aman in the Pune store could not fix my Sony headphones.",
    "The Taj Hotel in Delhi offered excellent service. Ankit was very helpful during my stay.",
    "Swiggy delivered my Domino’s Pizza hot and fresh. Ramesh was polite and professional.",
    "The Microsoft Windows update made my laptop noticeably faster.",
    "Zomato support handled my refund smoothly. Neha resolved the issue quickly.",
    "Apple disappointed me. My MacBook Air stopped charging and Chennai support was unhelpful.",
    "Reliance Digital customer service was poor. Aman in the Pune store could not fix my Sony headphones.",
    "Uber support was terrible in Hyderabad after my driver canceled the ride.",
    "My Myntra order never arrived. The Nike shoes were missing from the package.",
    "Indigo Airlines delayed my Mumbai to Delhi flight without any clear communication.",
    "The Dell Inspiron laptop overheats often and Dell support in Noida was ineffective."
]
    analyzer = FeedbackAnalyzer()
    analyze_result = analyzer.analyze_all(sample_reviews)
    print_results(analyze_result)

# Streamlit or Gradio ---> python Frontend Framework