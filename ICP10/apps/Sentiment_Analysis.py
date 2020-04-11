import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews


def extract_features(word_list):
    return dict([(word, True) for word in word_list])


def perform_sentiment_analysis(threshold_factor):
    print("Split : - train {} and test {}".format(threshold_factor * 100, 100 - threshold_factor * 100))
    # Load positive and negative reviews
    positive_fileids = movie_reviews.fileids('pos')
    negative_fileids = movie_reviews.fileids('neg')
    features_positive = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Positive') for f in positive_fileids]
    features_negative = [(extract_features(movie_reviews.words(fileids=[f])),
                          'Negative') for f in negative_fileids]
    # Split the data into train and test (80/20)
    threshold_positive = int(threshold_factor * len(features_positive))
    threshold_negative = int(threshold_factor * len(features_negative))

    features_train = features_positive[:threshold_positive] + features_negative[:threshold_negative]
    features_test = features_positive[threshold_positive:] + features_negative[threshold_negative:]
    print("\nNumber of training datapoints:", len(features_train))
    print("Number of test datapoints:", len(features_test))

    # Train a Naive Bayes classifier
    classifier = NaiveBayesClassifier.train(features_train)
    print("\nAccuracy of the classifier:", nltk.classify.util.accuracy(classifier, features_test))

    print("\nTop 10 most informative words:")
    for item in classifier.most_informative_features()[:10]:
        print(item[0])
    return classifier


def validate(classifier, input_reviews):
    print("\nPredictions:")
    for review in input_reviews:
        print("\nReview:", review)
        probdist = classifier.prob_classify(extract_features(review.split()))
        pred_sentiment = probdist.max()
        print("Predicted sentiment:", pred_sentiment)
        print("Probability:", round(probdist.prob(pred_sentiment), 2))


def get_input_reviews_basic():
    input_reviews = [
        "It is an amazing movie",
        "This is a dull movie. I would never recommend it to anyone.",
        "The cinematography is pretty great in this movie",
        "The direction was terrible and the story was all over the place"
    ]
    return input_reviews


def get_input_reviews_more_complex():
    input_reviews = [
        "Sunless Shadows is such a brilliant documentary because it treats the subject matter with the gravitas it needs. To accomplish this, Oskouei uses an intimate observational lens and his excellent relationship with the subjects.",
        "To have earned the trust of this community for a documentary film was significant enough, and the finished result proves to be really important. Simply put, Songs of Repression is one of the most compelling documentaries of the year",
        "this is a really bad movie, but the climax made it more than worth watching, its simply brilliant.",
        "For a film where you want to look away from the screen almost the whole time, Caught in the Net is impossibly gripping."
    ]
    return input_reviews


if __name__ == '__main__':
    trained_classifier = perform_sentiment_analysis(0.8)
    validate(trained_classifier, get_input_reviews_basic())
    validate(trained_classifier, get_input_reviews_more_complex())
    trained_classifier = perform_sentiment_analysis(0.7)
    validate(trained_classifier, get_input_reviews_basic())
    validate(trained_classifier, get_input_reviews_more_complex())
    trained_classifier = perform_sentiment_analysis(0.6)
    validate(trained_classifier, get_input_reviews_basic())
    validate(trained_classifier, get_input_reviews_more_complex())
    trained_classifier = perform_sentiment_analysis(0.5)
    validate(trained_classifier, get_input_reviews_basic())
    validate(trained_classifier, get_input_reviews_more_complex())
