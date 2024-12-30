import pandas as pd
#Define function for load datasets
def load_save_data(load_file_path, save_file_path):
    #read csv
    df= pd.read_csv(load_file_path)
    #select target column
    df=df[['message','label']]
    #droping some ham rows
    idx= df[df.label=='ham'].index[:1825]
    df.drop(idx,inplace=True)
    #shufling the data frame
    df = df.sample(frac=1)
    df.reset_index(inplace=True,drop=True)
    #save the data frame
    df.to_csv(save_file_path, index=False)
    print(f"The file save to: {save_file_path}")

load_save_data(
    load_file_path="./datasets/data.csv",
    save_file_path="./datasets/SPAM_DATASETS.csv"
)
