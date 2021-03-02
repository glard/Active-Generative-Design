import pandas as pd
import requests

df_candidate = pd.read_csv('./candidate_mpid_exp3_bo.csv')
df_us = df_candidate[['mpid']]
for mpid in df_candidate['mpid']:
    # download cif
    url = 'https://materialsproject.org/materials/' + str(mpid) + '/cif?type=primitive&download=true'
    r = requests.get(url, allow_redirects=True)  # to get content after redirection
    cif_url = r.url # 'https://xxxx.cif'
    print(cif_url)
    file_path = './CIF_exp3_bo/' + str(mpid) + '.cif'
    with open(file_path, 'wb') as f:
        f.write(r.content)
