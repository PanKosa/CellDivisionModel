#!/usr/bin/env python

import numpy as np


class Cell:
    def callculateMassGain(self,dt,T,growthRate):
        masses = np.zeros(round(T/dt)+1)
        masses[0] = self.m
        
        for i in range(len(masses)-1):
            masses[i+1] = masses[i] + growthRate(masses[i])*dt
                
        return masses
    
    def callculateDevisionCum(self,dt,T,devisionRate):
        cumul = np.zeros(round(T/dt)+3)
        cumul[0] = 1
        
        for i in range(len(self.masses)):
            #print("#####")
            #print(cumul[i])
            #print(self.masses[i])
            cumul[i+1] = max(0,cumul[i] - devisionRate(self.masses[i])*cumul[i]*dt)
        
        cumul = 1 - cumul
        
        prob = np.diff(cumul)
        
        np.append(prob, max(0, 1-sum(prob)))
                
        return prob
    
    def choseDevisionTime(self,dt,t,T):
        times = np.arange(round((T-t)/dt)+2)
        times = times*dt
        tt = np.random.choice(times, 1, p = self.prob)
        return tt
    def __init__(self, m, growthRate, devisionRate,dt,t,T):
        self.m = m
        self.it = 0
        self.r = 1
        self.masses = self.callculateMassGain(dt,T-t,growthRate)
        self.prob  = self.callculateDevisionCum(dt,T-t,devisionRate)
        self.devisionTime = self.choseDevisionTime(dt,t,T)
        #self.devisionTime =  np.random.uniform(0.5, 2)
        self.age = 0
        self.markedForDevision = 0
    def update(self,dt):
        self.it +=1
        self.age += dt
        self.m = self.masses[self.it]
        if self.age >= self.devisionTime:
            self.markedForDevision = 1
    def CellAfterDevision(self,growthRate,devisionRate,dt,t,T):
        self.age = 0.0
        self.m = self.m/2.0
        self.it = 0
        self.masses = self.callculateMassGain(dt,T-t,growthRate)
        self.prob   = self.callculateDevisionCum(dt,T-t,devisionRate)
        self.markedForDevision = 0  
        self.devisionTime = self.choseDevisionTime(dt,t,T)           