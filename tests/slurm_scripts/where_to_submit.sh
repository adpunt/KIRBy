#!/bin/bash
# One-shot: everything needed to decide WHERE (account) and WHICH PARTITION to
# submit for fastest scheduling right now. Read-only. No jobs touched.
U=scat9264
echo "########## 1. ALLOWED account / partition / qos ##########"
sacctmgr -n show assoc user=$U format=account%20,partition%15,qos%30
echo; echo "########## 2. FAIR-SHARE per account (higher FairShare = sooner) ##########"
sshare -u $U -o Account,User,RawShares,NormShares,EffectvUsage,FairShare
echo; echo "########## 3. PARTITIONS: state, wall limit, idle nodes ##########"
sinfo -o "%.15P %.6a %.12l %.6D %.6t %C" | grep -v -i drain
echo; echo "########## 4. YOUR jobs (running/pending) ##########"
squeue -u $U -o "%.12i %.10P %.9T %.12M %.5D %R"
echo "--- your job counts by state ---"
squeue -h -u $U -o "%T" | sort | uniq -c
echo; echo "########## 5. QUEUE CONTENTION: pending depth by partition ##########"
squeue -h -t PENDING -o "%P" | sort | uniq -c | sort -rn
