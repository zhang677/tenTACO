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
    elif libraries == 13:
        df_names = ['taco', 'tune0', 'tune1', 'tune2', 'tune3', 'tune4', 'tune5', 'tune6', 'tune7', 'tune8', 'tune9', 'tune10', 'tune11']
    elif libraries == 20:
        df_names = ['tune12', 'tune13', 'tune14', 'tune15', 'tune16', 'tune17', 'tune18', 'tune19', 'tune20', 'tune21', 'tune22', 'tune23', 'tune24', 'tune25', 'tune26', 'tune27', 'tune28', 'tune29', 'tune30', 'tune31']
    elif libraries == 32:
        df_names = ['ebsr0', 'ebsr1', 'ebsr2', 'ebsr3', 'ebsr4', 'ebsr5', 'ebsr6', 'ebsr7', 'ebsr8', 'ebsr9', 'ebsr10', 'ebsr11', 'ebsr12', 'ebsr13', 'ebsr14', 'ebsr15', 'ebsr16', 'ebsr17', 'ebsr18', 'ebsr19', 'ebsr20', 'ebsr21', 'ebsr22', 'ebsr23', 'ebsr24', 'ebsr25', 'ebsr26', 'ebsr27', 'ebsr28', 'ebsr29', 'ebsr30', 'ebsr31']
    structure(libraries, df_names, info, trials)


    
