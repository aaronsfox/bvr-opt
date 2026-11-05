import numpy as np, glob, sys
sys.path.insert(0,'.')
A='/mnt/user-data/attachments/de80d863-0bfc-5776-b7cf-80da25e9c8d8/'
files={'1stMet':'0fb7b310-1stMet_Tracking.tra','1stPhal':'66799824-1stPhal_Tracking.tra','2ndMet':'7db8de1d-2ndMet_Tracking.tra','2ndPhal':'3594c906-2ndPhal_Initial.tra'}
def load(n):
    return np.genfromtxt(A+files[n],delimiter=',')
