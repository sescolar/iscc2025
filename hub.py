# WPT: Wireless Power Transmission 
# Sole - 06/02/2025

import numpy as np
import math
import random 
import time
import matplotlib.pyplot as plt

# Constants for WPT paper
N_DEVICES = 6 
MAX_X = 100
MAX_Y = 100
ALPHA = (10 ** -2)

# Hub is located at (0,0) 
coordinates = [{'x': 0, 'y': 0, 'distance':0 } for _ in range(N_DEVICES)]

# Constants from previous works
HOUR_DIVIDE = 1 #2 3,4
K = (24 * HOUR_DIVIDE)
N_TASKS = 10
MAX_TASK_QL = 10
P = 1

#List of tasks for hub
tasks = [{'c_mAh': 0, 'q_perc': 0, 'wpt': 0} for _ in range(N_TASKS)]

#List of tasks for hub
N_TASKS_DEVICES = 2 
tasksDevices = [
    {
        'distance': 0,
        'tasks': [{'c_mAh': 0, 'q_perc': 0} for _ in range(N_TASKS_DEVICES)]
    }
    for _ in range(N_DEVICES)
]   


# Constants for consumption
ACTIVE_SYSTEM_CONSUMPTION = 124
IDLE_SYSTEM_CONSUMPTION  = 22 #idle is higher, we must check it
SEED = 124


# Battery constants
B_INIT = 1800
BMIN = 160
BMAX = 2000
BATTERY_SAMPLING = (int((BMAX-BMIN)*P))          # maximum = BMAX-BMIN
mAh_per_lvl = ((float)(BMAX-BMIN)/BATTERY_SAMPLING)
#for Carfagna
MAX_QUALITY_LVL = (int(100*P))     # maximum = 100 (percentage)


# Panel production
SEED = 124
VARIATION = 1
MAX_OVERPRODUCTION = 20
MAX_UNDERPRODUCTION = 40
SUNSET = 19
SUNRISE = 8
slot_duration_percentage = 24.0 / K  # K must be defined elsewhere in the code

# Hourly Energy harvested                                (October)
E_h = [0, 0, 0, 0, 0, 0, 0, 0, 19, 110, 224, 285, 335, 350, 331, 283, 134, 20, 18, 8, 0, 0, 0, 0]
E_h_v = [0] * 24  # Hourly Energy harvested varied
E_s_mAh = [0] * K


# max_task_ql quality level

B = np.zeros((2,K*MAX_TASK_QL+1))
S = np.zeros((K,K*MAX_TASK_QL+1))

def initialize_energy_harvested():
    total_energy_harvested = 0
    for index in range(24):
        if (SUNRISE <= index) and (index <= SUNSET):
            if not VARIATION:
                coin = 2
            else:
                coin = random.randint(0,2)
            match coin: 
                case 0 :
                    variation = 100 + (random.randint(0,MAX_OVERPRODUCTION))
                case 1 :
                    variation = 100 - (random.randint(0,MAX_UNDERPRODUCTION))
                case 2 :
                    variation = 100
            E_h_v[index] = min(int(E_h[index] * float(variation) / 100), 520)  # 520 es el máximo del panel
        #if K>24 we must spread the production between K/24 slots
        for j in range(0, K // 24 if K >= 24 else 0):
            E_s_mAh[(index * K // 24) + j] = int(E_h_v[index] * slot_duration_percentage)
        total_energy_harvested += E_h[index]
    if K > 24:
        scaled_sum = 0
        for index in range(24):
            for j in range(K // 24):
                E_s_mAh[(index * K // 24) + j] = int(E_h[index] * slot_duration_percentage)
                scaled_sum += E_s_mAh[(index * K // 24) + j]
        remaining_energy = total_energy_harvested - scaled_sum
        for j in range(K // 2, K):
            if remaining_energy > 0:
                E_s_mAh[j] += 1
                remaining_energy -= 1
    elif K < 24:
        scale_factor = 24 // K
        for index in range(K):
            E_s_mAh[index] = 0
            for j in range(index * scale_factor, (index + 1) * scale_factor):
                E_s_mAh[index] += E_h[j]

def check(K,S,Bstart,Bmin,Bmax,E,Tasks,debug=False):
    b = Bstart
    q = 0
    for i in range(K):
        bnew = min(b+E[i]-Tasks[S[i]].cost,Bmax)
        q += Tasks[S[i]].quality
        if debug: print("%2d :  %4d - %3d + %3d = %4d : %4d" %
        (S[i],b,Tasks[S[i]].cost,E[i],bnew,q))
        b = bnew
        if (b < Bmin): return 0
    if (b < Bstart):
        return 0
    return sum(Tasks[s].quality for s in S)

def ScheduleClassic(slots,Bstart,Bmin,Bmax,Eprod,Tasks):
    """
    slots = K = number of slots in a day
    Battery: Bmin < Bstart < Bmax 
    PanelProducton:  Eprod | len(Eprod) == slots
    Tasks: ordered from lowest cost,quality to greatest
    """
    # basic checks
    assert(len(Eprod) == slots)
    assert(Bmin < Bstart < Bmax)    
    M = np.zeros( (slots,Bmax+1), dtype=int)
    I = np.zeros( (slots,Bmax+1), dtype=int)
    for i in range(slots-1,-1,-1):
        for B in range(0,Bmax+1):
            qmax = -100
            idmax = 0
            if (i == slots-1):
                for t,task in enumerate(Tasks):
                    if (B-task['c_mAh']+Eprod[i] >= Bstart and task['q_perc'] > qmax):
                        qmax = task['q_perc']
                        idmax = t+1
            else:
                for t,task in enumerate(Tasks):
                    Bprime = min(B-task['c_mAh']+Eprod[i],Bmax)
                    if (Bprime >= Bmin):
                        q = M[i+1][Bprime]
                        if (q == 0): continue
                        if (q + task['q_perc'] > qmax):
                            qmax = q + task['q_perc']
                            idmax = t+1
            M[i][B] =  qmax if qmax != -100 else 0
            I[i][B] = idmax
    S = [0]*K
    B = Bstart
    for i in range(K):
        S[i] = I[i][B]-1
        if (S[i] < 0): return (S,0)
        B = min(B + Eprod[i] - Tasks[ S[i] ]['c_mAh'],Bmax)
        assert(B >= Bmin)
    assert(B >= Bstart)
    return(S,sum(Tasks[s]['q_perc'] for s in S))



def solution(tasks,upperbound):
    print("upperbound = ",upperbound)
    print(B[(K-1)%2][:])
    while upperbound >= 0 and (B[(K-1)%2][upperbound] < B_INIT or B[(K-1)%2][upperbound] == 0):
        print(B[(K-1)%2][upperbound])
        upperbound -= 1
    print("upperbound = ",upperbound)
    if upperbound < 0:
        return -1
    t = upperbound
    # note schedule use integers from 0 to 255
    schedule = np.zeros(K,dtype=np.int8)
    for s in range(K-1,-1,-1):
        schedule[s] = int(S[s][t])
        t -= tasks[ schedule[s]-1 ].quality
    print(schedule-1)
    return schedule

def ScheduleCarfagna(max_tax_ql,tasks,E) -> int:
    maxquality_previousslot = 0
    maxquality_currentslot = 0

    for k in range(K):
        print("."*40 + str(k) + "."*40)
        maxquality_currentslot = -1
        if (k == 0):
            print("iterations = ",max_tax_ql)
            for level in range(max_tax_ql):
                #print("level = ",level,end=", ")
                currentBmax = 0
                idMax = 0
                for index,t in enumerate(tasks):
                    Br = B_INIT + E[0] - t.cost;
                    if  t.quality == level and Br >= BMIN and Br >= currentBmax:
                        currentBmax = Br
                        idMax = index+1
                        maxquality_currentslot = level
                B[0][level] = min(currentBmax,BMAX)
                S[0][level] = idMax
                #print(B[0][level],", ",int(S[0][level]))
        else:
            print("iterations = ",maxquality_previousslot+max_tax_ql+1)
            for level in range(maxquality_previousslot+max_tax_ql+1):
                #print(level)
                currentBmax = 0
                idMax = 0
                for index,t in enumerate(tasks):
                     Bprec = B[(k-1)%2][level-t.quality]
                     Br = Bprec + E[k] - t.cost
                     if level >= t.quality and level - t.quality <= maxquality_previousslot and \
                        Bprec != 0 and Br >= BMIN and Br > currentBmax:
                        currentBmax = Br;
                        idMax = index+1;
                        maxquality_currentslot = level
                B[k%2][level] = min(currentBmax,BMAX);
                S[k][level] = idMax;
                #print(B[k%2][level],", ",int(S[k][level]))
        maxquality_previousslot = maxquality_currentslot
        if maxquality_previousslot == -1:
            return -1
    return maxquality_currentslot

def generate_tasks_devices():
    previous_distance = 0
    tasks_end = [{'c_mAh': 1, 'q_perc': 0},     #t0
                 {'c_mAh': 100, 'q_perc': 100}] #th

    for device_index, device_tasks in enumerate(tasksDevices):
        while True:
            x = random.uniform(-MAX_X, MAX_X)
            y = random.uniform(-MAX_Y, MAX_Y)
            coordinates[device_index]['x']=x
            coordinates[device_index]['y']=y
            distance = math.sqrt(abs(x)**2 + abs(y)**2)
            if distance > previous_distance:
                device_tasks['distance'] = distance 
                previous_distance = distance
                break

        for task_index,task in enumerate(device_tasks['tasks']):
            task['c_mAh'] = tasks_end[task_index]['c_mAh']
            task['q_perc'] = tasks_end[task_index]['q_perc']

def generate_tasks():
    tasks[0]['c_mAh'] = 1
    tasks[0]['q_perc'] = 1
    tasks[0]['wpt'] = 1

    for i in range(1, N_TASKS):
        tasks[i]['c_mAh'] = math.ceil((((i - 1.0) / 10.0) * ACTIVE_SYSTEM_CONSUMPTION + (1 - ((i - 1.0) / 10.0)) * IDLE_SYSTEM_CONSUMPTION) * slot_duration_percentage)

    scale = 100.0 / tasks[N_TASKS - 1]['c_mAh']
    for i in range(N_TASKS):
        tasks[i]['q_perc'] = math.ceil(tasks[i]['c_mAh'] * scale)

    scale_wpt = 20 / tasks[N_TASKS - 1]['c_mAh']
    for i in range(N_TASKS):
        tasks[i]['wpt'] = math.ceil(tasks[i]['c_mAh'] * scale_wpt)

def compute_quality(S):
    qavg = 0
    for i in range(K):
        if (S[i] < N_DEVICES):
            qavg = qavg + (tasks[S[i]]['q_perc']*((S[i]+1)/N_DEVICES)) 
        else:
            qavg = qavg + (tasks[S[i]]['q_perc']) 
    return  qavg

def scheduling_devices(S):
    Sdev = [[{'t': 0, 'q':0} for _ in range(K)] for _ in range(N_DEVICES)]
    Qdev = [0.0] * N_DEVICES
    for j in range(N_DEVICES):
        print("Schedule of end device: %d" % j)
        for i in range(K):
            if (j <= S[i]):
                Sdev[j][i]['t'] = 1
                Sdev[j][i]['q'] = tasksDevices[j]['tasks'][1]['q_perc'] 
            else:
                Sdev[j][i]['t'] = 0
                Sdev[j][i]['q'] = tasksDevices[j]['tasks'][0]['q_perc']
            Qdev[j] = Qdev[j] + Sdev[j][i]['q']
            if (i<K-1):
                print("%d, " % Sdev[j][i]['t'], end='')
            else:
                print("%d" % Sdev[j][i]['t'], end='')
        print("-- quality = %f " % (Qdev[j]/K))

    print("\n#----------------------------------------------\n")

def scheduling():
    Shub,Qhub=ScheduleClassic(K,B_INIT, BMIN,BMAX, E_s_mAh, tasks)
    q=compute_quality(Shub)
    print("Scheduling, Quality for hub")
    print(Shub,end='')
    print(" , %d - Avg(Q): %f" % (Qhub,q))

    print("\n#----------------------------------------------\n")
    scheduling_devices(Shub)
    return(Shub,Qhub)

#def compute_scheduling_devices(S,Q):
    #for n in range(N_DEVICES):
    #    S[n] = [0] * K
    #    for i in range(K):
    #        if (S[i]
    #        S[n][i]

        
def plot_network():
    # Extraer las coordenadas x e y
    x_values = [coordinate['x'] for coordinate in coordinates]
    y_values = [coordinate['y'] for coordinate in coordinates]
    plt.figure(figsize=(8,6))
    plt.scatter(x_values,y_values, c='red', marker='o', label="End Devices")
    plt.scatter(0, 0, c='blue', marker='o', s=100, label="Hub")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Network star for WPT")
    plt.legend()
    plt.grid(True)

    # Mostrar el gráfico
    #plt.show()
    plt.savefig("example-net.png")


def print_parameters():
    index = 0
    buffer = S  # Assuming S is defined elsewhere in the code
    print("\n#----------------------------------------------\n")
    print("K = %d" % K)
    print("N = %d" % N_TASKS)
    print("n = %d" % N_DEVICES)
    print("BMIN = %d" % BMIN)
    print("BINIT = %d" % B_INIT)
    print("BMAX = %d" % BMAX)
    print("BSAMPLING = %d" % BATTERY_SAMPLING)
    print("EnergyForLevel = %d" % mAh_per_lvl)
    print("MAX_QUALITY_LVL = %d" % MAX_QUALITY_LVL)


    print("\nc_i = [",end='')
    for index in range(N_TASKS):
        buffer = f"{tasks[index]['c_mAh']:3d}{',' if index < N_TASKS - 1 else ']'}"
        print(buffer,end='')

    print("\nq_i = [",end='')
    for index in range(N_TASKS):
        buffer = f"{tasks[index]['q_perc']:3d}{',' if index < N_TASKS - 1 else ']'}"
        print(buffer,end='')

    print("\nwpt_i = [",end='')
    for index in range(N_TASKS):
        buffer = f"{tasks[index]['wpt']:3d}{',' if index < N_TASKS - 1 else ']'}"
        print(buffer,end='')

    print("\ne_i = [",end='')
    for index in range(24):
        buffer = f"{E_h[index]:3d}{',' if index < 24 - 1 else ']'}"
        print(buffer,end='')

    print("\nE_i = [",end='')
    for index in range(K):
        buffer = f"{E_s_mAh[index]:3d}{',' if index < K - 1 else ']'}"
        if (index + 1) % 25 == 0:
            print("\n")
        print(buffer,end='')

    print("\n#----------------------------------------------\n")

    for device_index, device_tasks in enumerate(tasksDevices):
        print(f"Device {device_index}:")
        print(f"  Distance {device_tasks['distance']:.4f}, ({coordinates[device_index]['x']:.4f}, {coordinates[device_index]['y']:.4f})")
        for task_index, task in enumerate(device_tasks['tasks']):
            print(f"  Task {task_index}: c_mAh = {task['c_mAh']}, q_perc = {task['q_perc']}")
        print()  

    print("\n#----------------------------------------------\n")


if __name__ == "__main__":
    random.seed(time.time())  # Semilla basada en el tiempo actual
    generate_tasks()
    initialize_energy_harvested()
    generate_tasks_devices()
    plot_network()
    print_parameters()
    Shub,Q=scheduling()
    #compute_scheduling_devices(Shub,Qhub)


