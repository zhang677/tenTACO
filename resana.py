from re import L
import sys
import pandas as pd
# python resana.py ./matrix_results/spmm_csr_gpu.csv res1020.csv
# scp zhr.eva7:/home/nfs_data/zhanggh/mytaco/learn-taco/zghshared/res1020.csv /Users/zhang/Desktop
def structure(libraries, df_names, info, trials):
    cnt = 0
    tmp = 0
    tmps = []
    setloop = 0
    dfs = [[] for i in range(libraries)]
    dataset = []
    for i in range(lines):
        tmps.append(info.loc[i].time)
        tmp = tmp + info.loc[i].time
        if(cnt==trials-1):
            setloop = setloop%libraries
            cnt = 0
            tmp = tmp / trials
            tmp = min(tmps)
            if(setloop == 0):
                dataset.append(info.loc[i].tensor)
            dfs[setloop].append(round(tmp,4))
            tmp = 0
            tmps = []
            setloop = setloop + 1
        else:
            cnt = cnt + 1

    df_dict = {'dataset':dataset}
    for i in range(libraries):
        df_dict[df_names[i]] = dfs[i]
    outf = pd.DataFrame(df_dict)
    outf.to_csv(outfile,index=False,sep=',')


if __name__ == '__main__':
    infofile = sys.argv[1]
    outfile = sys.argv[2]
    libraries = int(sys.argv[3])
    info = pd.read_csv(infofile)
    lines = info.shape[0]
    trials = 25
    accu = 4 # round(4)
    if libraries == 3:
        df_names = ['taco', 'eb-pr', 'eb-sr']
    structure(libraries, df_names, info, trials)


    
