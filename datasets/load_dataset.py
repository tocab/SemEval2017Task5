import pandas as pd
import sys
import json


def load_track1_full_data(path):
    # Load track 1 data

    # Create new pandas dataframes
    try:
        with open(path) as file:
            data = json.load(file)
    except:
        print("File not found")
        sys.exit(1)

    # Load to pandas dataframe and normalize json
    df_track1 = pd.io.json.json_normalize(data)

    # Get only required columns
    df_track1 = df_track1[
        ["id", "cashtag", "spans", "text", "message.body", "sentiment score", "source",
         "message.entities.sentiment.basic"]]

    # Divide data in stocktwits/twitter and empty
    df_track1_twitter = df_track1[df_track1["text"].isnull() == False]
    df_track1_stocktwits = df_track1[df_track1["message.body"].isnull() == False]
    df_track1_empty = df_track1[df_track1["message.body"].isnull() & df_track1["text"].isnull()]

    # Delete message.body because twitter text is in column "text"
    df_track1_twitter = df_track1_twitter.drop('message.body', 1)

    # Rename message.body to text
    df_track1_stocktwits = df_track1_stocktwits.drop('text', 1)
    df_track1_stocktwits.rename(columns={'message.body': 'text'}, inplace=True)

    # Delete message.body column in empty messages
    df_track1_empty = df_track1_empty.drop('message.body', 1)
    df_track1_empty["text"] = "||EMPTYTEXT||"

    # Concat the three DataFrames
    df_track1 = pd.concat([df_track1_twitter, df_track1_stocktwits, df_track1_empty], ignore_index=True)

    # Convert span lists to text
    df_track1["spans"] = df_track1.apply(list_to_text, axis="columns")

    # Write index in column
    df_track1["index"] = df_track1.index

    # make sentiment score numeric
    df_track1["sentiment score"] = pd.to_numeric(df_track1["sentiment score"])

    return df_track1


def load_track2_full_data(path):
    # Load track 2 data

    # Create new pandas dataframes
    try:
        with open(path) as file:
            data = json.load(file)
    except:
        print("File not found.")
        sys.exit()

    # Load to pandas dataframe and normalize json
    data = pd.io.json.json_normalize(data)

    data.rename(columns={'title': 'text', "sentiment": "sentiment score"}, inplace=True)

    # Write index in column
    data["index"] = data.index

    return data


def load_track1_test_data(path):
    # Load track 1 data

    # Create new pandas dataframes
    try:
        data = pd.read_json(path)
    except:
        print("File not found.")
        sys.exit()

    # Convert span lists to text
    data["spans"] = data.apply(all_to_string, axis="columns")

    # Create Column sentiment score
    data["sentiment score"] = 0.0

    # Write index in column
    data["index"] = data.index

    # Clean spans
    data["spans"] = data["spans"].replace(r';[$];', r' ', regex=True)

    return data


def load_track2_test_data(path):
    # Load track 1 data

    # Create new pandas dataframes
    try:
        data = pd.read_json(path)
    except:
        print("File not found")
        sys.exit(1)

    # Concat dataframes
    data.rename(columns={'title': 'text'}, inplace=True)

    # Create Column sentiment score
    data["sentiment score"] = 0.0

    # Write index in column
    data["index"] = data.index

    return data


def load_track1_data(path):
    # Load track 1 data

    # Create new pandas dataframes
    try:
        data = pd.read_json(path)
    except:
        print("File not found.")
        sys.exit(1)

    # Convert span lists to text
    data["spans"] = data.apply(list_to_text, axis="columns")

    # Write index in column
    data["index"] = data.index

    return data


def list_to_text(row):
    """
    :param row: Row of the dataframe that applies that function
    :return: Text out of list
    """
    text = ' '.join(row['spans'])
    return text


def all_to_string(row):
    text = str(row['spans'])
    return text
