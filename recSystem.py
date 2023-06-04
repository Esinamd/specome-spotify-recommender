import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.preprocessing import StandardScaler
from scipy import sparse
from datetime import datetime as time
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import normalize
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PowerTransformer
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
# import mlrose
from flask_cors import CORS
from flask import Flask, request, jsonify
import os

# ensuring correct path is always used with datasets
csvdir = os.path.dirname(os.path.abspath(__file__))
tracksPath = os.path.join(csvdir, 'spotify datasets', 'tracks.csv')
artistsPath = os.path.join(csvdir, 'spotify datasets', 'artists.csv')

app = Flask(__name__)
CORS(app)


@app.route("/Songs2", methods=['GET', 'POST'])
def pyPageSongs2():
    if request.method == 'POST':
        data = request.get_json()
        print(data)
        name = data["name"]
        number = int(data["number"])
        artist = data["artist"]

        listed = False
        playlistSongs = []
        playlistArtists = []

        while listed == False:
            try:
                cosine, ind, dataset = mainTracks()
                id = getSongId(name, artist, dataset)
                playlist = recommendSong(id, cosine,
                                         ind, dataset, number).values

                for p in playlist:
                    playlistSongs.append(p[0])
                    s = str(p[1])
                    s = s.replace('[', '')
                    s = s.replace(']', '')
                    s = s.replace('\'', '')
                    playlistArtists.append(s)

                print(playlist)
                listed = True
            except:
                print(
                    "There seems to be an error, please check your inputs or try another song")
                break

        if type(playlistArtists) is not list:
            playlistArtists = playlistArtists.tolist()
        elif type(playlistSongs) is not list:
            playlistSongs = playlistSongs.tolist()

        songDict = {'listed': listed, 'listSongs': playlistSongs,
                    'listArtists': playlistArtists}
        return jsonify(songDict)
    else:
        return jsonify({'message': "songs"})


@ app.route("/Artists2", methods=['GET', 'POST'])
def pyPageArtists2():
    if request.method == 'POST':
        data = request.get_json()
        name = data["name"]
        number = int(data["number"])

        listed = False
        playlist = []

        while listed == False:
            try:

                cosine, ind, dataset = mainArtists()
                id = getArtistId(name, dataset)
                playlist = recommendArtist(id, cosine,
                                           ind, dataset, number).values
                print(playlist)
                listed = True

            except:
                print(
                    "There seems to be an error, please check your inputs or try another artist")
                break

        if type(playlist) is not list:
            playlist = playlist.tolist()
        artDict = {'listed': listed, 'list': playlist}
        return jsonify(artDict)

    else:
        return jsonify({'message': "artists2"})


@ app.route("/SpotifyConnect", methods=['GET', 'POST'])
def pySpotifyConnect():
    if request.method == 'POST':
        data = request.get_json()
        features = data["features"]
        recType = data["type"]

        features = pd.DataFrame(features)

        if recType == "artists":
            features = features.drop(
                columns=["external_urls", "href", "images", "type", "uri", "followers"])
            dataset = pd.read_csv(artistsPath)
            cleanPreprocessing(dataset)
            resetIndex(dataset)
            dataset = dataset.drop(columns=["followers"])

            # dataset["genres"] = dataset["genres"].apply(lambda a: a.replace(
            #     '[', '').replace(']', '').replace('\'', '').split(",") if a != "[]" else [])

        elif recType == "songs":
            features = features.drop(columns=["loudness", "liveness", "type", "duration_ms",
                                              "key", "tempo", "uri", "track_href", "analysis_url", "time_signature"])

            dataset = pd.read_csv(tracksPath)
            cleanPreprocessing(dataset)
            resetIndex(dataset)
            dataset = dataset.drop(columns=["popularity", "loudness", "liveness", "duration_ms",
                                   "key", "tempo", "time_signature", "explicit", "release_date"])

        mmScaler = MinMaxScaler()
        mlb = MultiLabelBinarizer()
        if recType == "artists":
            encGenre = mlb.fit_transform(features["genres"])
            encodedFeatures = features.copy()
            encodedFeatures["genres"] = encGenre
            encodedFeatures = encodedFeatures.drop_duplicates()
            encodedFeatures = encodedFeatures.drop(columns=["id", "name"])
            print("cols", encodedFeatures.columns)

        elif recType == "songs":
            encodedFeatures = features.copy()
            encodedFeatures = encodedFeatures.drop(columns=["id"])

        scaledFeatures = np.array(mmScaler.fit_transform(encodedFeatures))
        simScores, indices = listSimScore(scaledFeatures, dataset, recType)
        playlist = recommendSpotify(
            simScores, indices, scaledFeatures, dataset, recType)

        playlist = playlist.values.tolist()

        # spotifyDict = {'list': playlist}
        return jsonify(playlist)
    else:
        return jsonify({'message': "SpotifyConnect"})


if __name__ == "__main__":
    print('oh hello')
    # time.sleep(5)
    app.run(host='127.0.0.1', port=5000)


def cleanPreprocessing(dataset):
    print("Cleaning...")
    # preprocessing spotify tracks and artists datasets
    print("inital size dataset ", len(dataset))
    # checking for missing values
    if dataset.isnull().values.any():
        # print(dataset.isnull().sum().sum(), "records with missing values")
        dataset.dropna(inplace=True)
        # print("remaining records:", len(dataset))
    else:
        # print("no missing records")
        pass
    # checking for duplicate records
    if dataset.duplicated().values.all():
        # print("no duplicate records")
        pass
    else:
        # print(dataset.duplicated().sum().sum(), "duplicate records")
        dataset.drop_duplicates(inplace=True)
        # print("records remainin:", len(dataset))

    return dataset


def dupTracks(datasetTracks):
    # checking for duplicate song tracks
    if datasetTracks.duplicated(subset=['name', 'artists']).values.all():
        # print("no duplicate song tracks")
        pass
    else:
        # print(datasetTracks.duplicated(
        # subset=['name', 'artists']).sum().sum(), "duplicate songs")
        datasetTracks.drop_duplicates(
            subset=['name', 'artists'], inplace=True)
        # print("records remaining in tracks:", len(datasetTracks))
    return datasetTracks


def resetIndex(dataset):
    print("size of dataset ", len(dataset))
    # reset the row index for datasets
    dataset.reset_index(drop=True, inplace=True)
    return dataset


def catTracks(dataset):
    print("Categorising...")
    # convert release date column to type timestamp
    dataset["release_date"] = pd.to_datetime(
        dataset["release_date"], errors="coerce")

    genre = []

    # maually classify 100,000 random tracks
    randomTracks = dataset.sample(n=100000).reset_index(drop=True)

    for i in range(len(randomTracks)):
        track = randomTracks.iloc[i]

        # pop music
        if track["popularity"] >= 65:
            genre.append("POP")
        # instrumental music
        elif track["instrumentalness"] >= 0.5 and track["speechiness"] < 0.1:
            genre.append("INSTRUMENTAL")
        # RnB music
        elif (225000 <= track["duration_ms"] <= 60 and 140 >= track["tempo"] >= 60 and track["release_date"].year >= 1940):
            genre.append("RNB")
        # rock music
        elif (track["energy"] >= 0.6 and 140 >= track["tempo"] >= 90 and track["danceability"] <= 0.5 and track["release_date"].year >= 1950):
            genre.append("ROCK")
        # Hiphop music
        elif (track["speechiness"] >= 0.25 and 115 >= track["tempo"] >= 85 and track["release_date"].year >= 1970):
            genre.append("HIPHOP")
        # DnB music
        elif (track["tempo"] >= 165 and track["speechiness"] < 0.2 and track["release_date"].year >= 1900):
            genre.append("DNB")
        # Latin music
        elif (100 >= track["tempo"] >= 80 and track["danceability"] >= 0.7 and track["energy"] >= 0.65):
            genre.append("LATIN")
        else:
            genre.append("ALT")

    randomTracks["genre"] = genre

    return randomTracks


def eraTracks(dataset):
    era = []
    releaseDates = dataset["release_date"]

    for releaseDate in releaseDates:
        year = int(releaseDate[:4])

        if year >= 2010:
            era.append("modern")
        elif 2000 <= year <= 2009:
            era.append("00s")
        elif 1990 <= year <= 1999:
            era.append("90s")
        elif 1980 <= year <= 1989:
            era.append("80s")
        elif 1970 <= year <= 1979:
            era.append("70s")
        elif 1960 <= year <= 1969:
            era.append("60s")
        elif 1950 <= year <= 1959:
            era.append("50s")
        else:
            era.append("none")

    dataset["era"] = era

    return dataset


def mlpTraining(dataset):
    # use songs with genre labels to train model
    trainingTracksDS = dataset.loc[dataset["genre"] != "ALT"]

    features = trainingTracksDS[["time_signature", "tempo", "valence", "liveness", "instrumentalness",
                                 "acousticness", "speechiness", "mode", "loudness", "key", "energy", "danceability", "explicit"]]
    # splitting data into training and testing (20%) sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, trainingTracksDS["genre"], test_size=0.2)

    # training MLP classifier neural network model
    mlp = MLPClassifier().fit(x_train, y_train)

    # accuracy score of the model
    accuracy = mlp.score(x_test, y_test)
    # print(accuracy)
    return x_test, y_test


def hyperparameterTuning(x_test, y_test):
    # hyperparameter tuning to improve accuracy

    # params of model
    params = {'activation': ['tanh'],
              'solver': ['adam'],
              'learning_rate': ['invscaling'],
              'hidden_layer_sizes': [(200,)]}

    # using gridsearch cv hyper parameter tool
    gridSearch = GridSearchCV(estimator=MLPClassifier(
        max_iter=500), param_grid=params, scoring="accuracy")
    # scaling and transforming data with power transformer
    transformedData = PowerTransformer().fit_transform(x_test)
    gridSearch.fit(transformedData, y_test)

    # storing the results of the model
    results = pd.DataFrame(gridSearch.cv_results_['params'])
    results['testScore'] = gridSearch.cv_results_['mean_test_score']

    # predicting on test dataset
    improvedPredict = gridSearch.predict(transformedData)

    # new improved accuracy
    improvedAccuracy = gridSearch.score(transformedData, y_test)
    # print("new accuracy is ", improvedAccuracy)
    return gridSearch


def logRegTraining(dataset):
    print("Training...")
    # use songs with genre labels to train model
    trainingTracksDS = dataset.loc[dataset["genre"] != "ALT"]

    # train with most important features
    features = trainingTracksDS[["popularity", "danceability", "energy", "speechiness",
                                 "instrumentalness", "valence", "tempo"]]

    # splitting data into training and testing (20%) sets
    x_train, x_test, y_train, y_test = train_test_split(
        features, trainingTracksDS["genre"], test_size=0.2)

    # model without optimisation from mlrose
    lg = LogisticRegression(
        random_state=0, solver="liblinear").fit(x_train, y_train)
    lgLabels = lg.predict(x_test)
    print("lg accuracy", accuracy_score(lgLabels, y_test))

    # cm = confusion_matrix(y_test, lgLabels)
    # display_matrix = ConfusionMatrixDisplay(cm).plot()
    # display_matrix.figure_.savefig("RecSysConfMatrix.png")

    return lg


def predictDataset(model, dataset):
    print("Predicting...")
    unlabelledTracks = dataset[["popularity", "danceability", "energy", "speechiness",
                                "instrumentalness", "valence", "tempo"]]
    genreLabels = model.predict(unlabelledTracks)
    dataset["genre"] = genreLabels

    return dataset


def encodeTrack(dataset):
    # encoding string genre value and era
    le = LabelEncoder()
    recGenre = le.fit_transform(dataset["genre"])
    recEra = le.fit_transform(dataset["era"])

    recDS = dataset.copy()

    recDS["genre"] = recGenre
    recDS["era"] = recEra

    # most important features when recommending songs
    recFeatures = ["popularity", "era", "genre"]
    # simFeatures = ["id", "name", "artists", "popularity", "era", "genre"]
    # recDS = recDS[simFeatures]

    return recDS, recFeatures


def encodeArtist(dataset):
    # encoding the name and genres columns
    le = LabelEncoder()
    # recNames = le.fit_transform(dataset["name"])
    recGenres = le.fit_transform(dataset["genres"])

    recDS = dataset.copy()

    recDS["genres"] = recGenres
    # recDS["name"] = recNames

    # most important features when recommending songs
    recFeatures = ["genres"]
    # simFeatures = ["id",  "followers", "genres", "name", "popularity"]

    return recDS, recFeatures


def scalingData(encodedDataset, features):
    print("Scaling...")

    # feature scaling and normalising the data
    scaler = MinMaxScaler()
    normalisedDS = np.array(scaler.fit_transform(
        encodedDataset[features]), dtype=np.float16)

    return normalisedDS


# def calculateCosSimBatch(batch, dataset):
#     # function calculating cosine sim for one batch
#     print("Batch similarity...")
#     # print("batch:", batch)
#     # print("dataset:", dataset)
#     dotProd = np.dot(batch, dataset.T)
#     batchMagnitude = np.sqrt(np.sum(batch ** 2, axis=1, keepdims=True))
#     datasetMagnitude = np.sqrt(np.sum(dataset ** 2, axis=1, keepdims=True))
#     # cosineSim = dotProd / (batchMagnitude * datasetMagnitude.T)

#     # handle 0 division
#     cosineSim = np.divide(dotProd, batchMagnitude*datasetMagnitude.T,
#                           out=np.zeros_like(dotProd), where=(batchMagnitude*datasetMagnitude.T) != 0).astype(np.float16)
#     return cosineSim


# def similarityScore(dataset, normDS):
#     print("Similarity score...")

#     batchSize = 500
#     nRows = normDS.shape[0]
#     nBatches = (nRows + batchSize - 1) // batchSize

#     cosine = []
#     ind = []
#     # initialise maximum dimension
#     maxDim = 0

#     for i in range(nBatches):
#         startInd = i*batchSize
#         endInd = startInd + batchSize
#         batch = normDS[startInd:endInd]
#         cosSimBatch = calculateCosSimBatch(batch, normDS)
#         cosine.append(cosSimBatch)

#         indexData = pd.DataFrame(dataset[startInd:endInd])
#         # index for current batch
#         batchIndex = pd.Series(
#             indexData.index, name="id", index=indexData["id"].str.lower())
#         ind.append(batchIndex)

#         # Update maximum dimension
#         maxDim = max(maxDim, batch.shape[1])

#     # Pad and concatenate the cosine similarity matrices
#     cosinePadded = []
#     for sim in cosine:
#         paddedSim = np.pad(
#             sim, ((0, 0), (0, maxDim - sim.shape[1])), mode='constant')
#         cosinePadded.append(paddedSim)
#     print("cos", len(cosine))

#     cosine = np.concatenate(cosinePadded, axis=0).astype(np.float16)

#     # Concatenate the batch indices
#     ind = pd.concat(ind)

#     print("ind", len(ind), "shape", ind.shape[0])
#     print("cos", len(cosine), "shape", cosine.shape)

#     # eliminate self similarity by setting diagonal to 0
#     np.fill_diagonal(cosine, 0)

#     # # replace 1.0(and equivalent) with 0.0 since it implies theyre the same song
#     # cosine = np.array(cosine)
#     # cosine[cosine > 0.99] = 0.0
#     # cosine = cosine.tolist()

#     return cosine, ind


def similarityScore(dataset, normDS):
    print("Similarity score...")
    batchSize = 250
    nRows = normDS.shape[0]
    nBatches = (nRows + batchSize - 1) // batchSize

    cosine = []
    ind = []
    # initialise maximum dimension
    maxDim = 0

    for i in range(nBatches):
        startInd = i*batchSize
        endInd = min((i+1)*batchSize, nRows)
        batchData = pd.DataFrame(normDS[startInd:endInd])
        indexData = pd.DataFrame(dataset[startInd:endInd])

        # cosine sim for current batch
        batchSim = cosine_similarity(batchData)

        # index for current batch
        batchIndex = pd.Series(
            indexData.index, name="id", index=indexData["id"].str.lower())

        cosine.append(batchSim)
        ind.append(batchIndex)

        # Update maximum dimension
        maxDim = max(maxDim, batchSim.shape[1])

    # Pad and concatenate the cosine similarity matrices
    cosinePadded = []
    for sim in cosine:
        paddedSim = np.pad(
            sim, ((0, 0), (0, maxDim - sim.shape[1])), mode='constant')
        cosinePadded.append(paddedSim)

    cosine = np.concatenate(cosinePadded, axis=0)

    # Concatenate the batch indices
    ind = pd.concat(ind)

    # eliminate self similarity by setting diagonal to 0
    np.fill_diagonal(cosine, 0)

    # replace 1.0(and equivalent) with 0.0 since it implies theyre the same song
    cosine = np.array(cosine)
    cosine[cosine > 0.99] = 0.0
    cosine = cosine.tolist()

    return cosine, ind


def listSimScore(spotifyList, dataset, recType):
    print("Spotify similarity score...")
    print("spot", spotifyList.shape)
    print("ds", dataset.shape)
    simScores = []
    indices = []

    batchData = pd.DataFrame(spotifyList)
    batchFeatures = pd.DataFrame(dataset)
    indices = pd.Series(
        batchFeatures.index, name="id", index=batchFeatures["id"].str.lower())

    if recType == "artists":
        mlb = MultiLabelBinarizer()
        encGenre = mlb.fit_transform(batchFeatures["genres"])
        # encodedFeatures = features.copy()
        batchFeatures["genres"] = encGenre
        batchFeatures = batchFeatures.drop_duplicates()

    # removing non numerical columns
    batchFeatures = batchFeatures.select_dtypes(include=[np.number])

    # Cosine similarity for current batch
    simScores = cosine_similarity(batchData, batchFeatures)

    # Eliminate self-similarity by setting the diagonal to 0
    np.fill_diagonal(simScores, 0)

    # Replace 1.0 (and equivalent) with 0.0 since it implies they're the same song
    simScores[simScores > 0.99] = 0.0

    return simScores, indices


def getSongId(songName, songArtist, dataset):
    print("songid...")
    ds = dataset.copy()
    ds["name"] = ds["name"].str.lower()
    # n = ds.loc[ds["name"].str.contains(songName, case=False)]
    n = ds.loc[ds["artists"].str.contains(songArtist, case=False)]
    # idRow = n.loc[ds["artists"].str.contains(songArtist, case=False)]
    idRow = n.loc[ds["name"].str.contains(songName, case=False)]
    id = idRow["id"].values
    id = str(id[0]).lower()

    return id


def getArtistId(artistName, dataset):
    print("artistid...")
    ds = dataset.copy()
    ds["name"] = ds["name"].str.lower()
    n = ds.loc[ds["name"].str.contains(artistName, case=False)]
    n = n.sort_values(by='popularity', ascending=False)
    id = n["id"].values
    id = str(id[0]).lower()
    print(id)

    return id


def recommendSong(song_id, model_type, ind, dataset, number):
    # inoput is song title and type of similarity model
    # output is a pandas series of recommended songs
    print("Recommending songs...")
    index = ind[song_id]
    print("chosen song", index)
    songList = list(enumerate(model_type[index]))
    similarityScore = sorted(songList, key=lambda x: x[1], reverse=True)
    # top recommended songs
    similarityScore = similarityScore[1: number + 1]
    print(similarityScore)
    topSongsInd = [i[0] for i in similarityScore]
    topSongs = dataset[["name", "artists"]].iloc[topSongsInd]
    print(topSongs)
    return topSongs


def recommendArtist(artistId, modelType, ind, dataset, number):
    print("Recommending artists...")
    index = ind[artistId]
    print("chosen artist", index)
    songList = list(enumerate(modelType[index]))
    simScore = sorted(songList, key=lambda x: x[1], reverse=True)
    simScore = simScore[1: number + 1]
    print(simScore)
    topArtInd = [i[0] for i in simScore]
    topArt = dataset["name"].iloc[topArtInd]
    print(topArt)
    return topArt


def recommendSpotify(simScores, indices, spotifyList, dataset, recType, N=10):
    print("Spotify recommendation...")
    print("sims shapes", simScores.shape)
    print("ind shaoe", indices.shape)
    recommended = []
    scores = []
    print("pop", len(indices))
    print(spotifyList)

    for i in range(len(spotifyList)):
        songIndex = indices[i]
        songSimScores = simScores[songIndex]
        topIndices = np.argsort(songSimScores)[::-1][:N]
        scores.append(topIndices)
        topSongs = dataset.iloc[topIndices]
        recommended.append(topSongs)

    recommended = pd.concat(recommended)
    scores = np.concatenate(scores)
    recommended["sim_scores"] = scores
    print("before", recommended.head(N))
    recommended = recommended.sort_values(by=["sim_scores"], ascending=False)
    print("after", recommended.head(N))

    if recType == "artists":
        recommended = recommended["name"].str.lower()
    elif recType == "songs":
        recommended["name"] = recommended["name"].str.lower()
        recommended["artists"] = recommended["artists"].str.lower()
        recommended = recommended[["name", "artists"]]

    recommended = recommended.drop_duplicates()
    recommended = recommended.head(N)
    print("spotify playlist", recommended)
    return recommended


def mainTracks():
    tracksDS = pd.read_csv(tracksPath)

    cleanPreprocessing(tracksDS)
    # resetIndex(tracksDS)
    # dropZeroPop(tracksDS)
    dupTracks(tracksDS)
    resetIndex(tracksDS)
    eraTracks(tracksDS)
    catDS = catTracks(tracksDS)
    lr = logRegTraining(catDS)
    # x_test, y_test = mlpTraining(catDS)
    # mlp = hyperparameterTuning(x_test, y_test)
    # predictDataset(mlp, tracksDS)
    predictDataset(lr, tracksDS)
    resetIndex(tracksDS)
    ds, features = encodeTrack(tracksDS)
    normDS = scalingData(ds, features)
    cosine, ind = similarityScore(tracksDS, normDS)

    return cosine, ind, tracksDS


def mainArtists():
    artistsDS = pd.read_csv(artistsPath)

    cleanPreprocessing(artistsDS)
    # resetIndex(artistsDS)
    # dropZeroPop(artistsDS)
    resetIndex(artistsDS)
    ds, features = encodeArtist(artistsDS)
    normDS = scalingData(ds, features)
    cosine, ind = similarityScore(artistsDS, normDS)

    return cosine, ind, artistsDS
