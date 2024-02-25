import pandas as pd
import json
import os

# reading file
def read_file(file_path: str="data_policyqa/train.json"):
    with open(file_path, 'r') as file:
        data = json.load(file)
    return data

# return a list of policy titles
def create_policy_titles(data: dict) -> list:
    titles = []
    for sample in data['data']:
        titles.append(sample['title'])
    return titles

# return the corresponding file name
def creat_file_names(policy_titles: list, policy_df: pd.DataFrame) -> list:
    file_names = []
    for title in policy_titles:
        find_title = policy_df['Policy URL'].apply(lambda x: title in x)
        id = policy_df[find_title]['Policy UID'].values
        # Check that the ID exists in the dataset and is unique
        if len(id) != 1:
            print(f"Warning: policy title {title} returned {len(id)} id matches. It was skipped.")
            file_names.append(None)
            continue
        id = id[0]
        file_name = str(id)+'_'+title+'.html'
        file_names.append(file_name)
    return file_names

# check whether the files exist in the targeted directory
def check_files_exist(files: list, dir:str = "./data_opp-115/OPP-115/sanitized_policies/" ):
    for file in files:
        if file:
            if not os.path.isfile(dir+file):
                print(f"{file} was not found in the intended directory.")
    print("Checking Files existance Finished!")

# Remove Context, Index and Summary 
def remove_contex_summary(data: dict) -> dict:
    out = {}
    out['version'] = data['version']
    out['data'] = [{}]*len(data['data'])

    for i in range(len(data['data'])):
        out['data'][i]['file_name']=''
        out['data'][i]['policy_title']=data['data'][i]['title']
        qas=[]
        for paragraph in data['data'][i]['paragraphs']:
            qas.extend(paragraph['qas'])
        out['data'][i]['qas'] = qas
    return out

# Add files names to the data
def add_file_names(files: list, data: dict):
    if len(files) != len(data['data']):
        print(f"There is a mismatch between the length of list of names and data passed")
        return -1
    for i in range(len(files)):
        data['data'][i]['file_name'] = files[i]
    return 0

# write out the json file
def write_json(data: dict, file_name: str= "prepared_data/train.json"):
    with open(file_name, 'w') as file:
        json.dump(data, file, indent=4)


def creat_data(path_to_read: str, path_to_write: str, policy_df: pd.DataFrame):
    '''
    create a Q&A dataset with reference to the corresponding Privacy Policy files
    from two data sources: PolicyQA and OPP-115 Corpus

    Parameters:
    path_to_read (str): path to PolicyQA train.json/dev.json/test.json file .
    path_to_write (str): path to output json file
    policy_df (DataFrame): DataFrame with data on all policies covered by OPP-115  
    '''
    # create train data
    data = read_file(path_to_read)
    titles = create_policy_titles(data)
    file_names = creat_file_names(titles, policy_df)

    # Check that the files actually exists
    check_files_exist(file_names)

    # clean data
    clean_data = remove_contex_summary(data) # comment this line to create raw data
    add_file_names(file_names, clean_data)
    write_json(data, path_to_write)
        
if __name__ == "__main__":
    policy_df = pd.read_csv("data_opp-115/OPP-115/documentation/policies_opp115.csv")

    # create train data 
    creat_data("data_policyqa/train.json", "prepared_data/qa/train.json")
    # create dev data 
    creat_data("data_policyqa/dev.json", "prepared_data/qa/dev.json")
    # create test data 
    creat_data("data_policyqa/test.json", "prepared_data/qa/test.json")