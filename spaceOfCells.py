#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.axes_grid1 import make_axes_locatable

from cell import *

class SpaceOfCells:
    def __init__(self, m , dimX, dimY, growthRate, devisionRate, dt, T):
        """
        inicjujemy uklad pojedyncza komorka 
        m - masa poczatkowa
        dimX - rozmiar macierzy w ktorej beda zachodzic podzialy komorek
        dimX - rozmiar macierzy w ktorej beda zachodzic podzialy komorek
        """ 
        self.growthRate = growthRate
        self.devisionRate = devisionRate
        self.dt = dt
        self.T = T
        self.t = 0
        self.CellMatrixXdim = dimX
        self.CellMatrixYdim = dimY
        self.CellSize = m
        self.CellMatrix = np.empty( (dimX,dimY), dtype=Cell)
        self.CellMatrix[dimX//2, dimY//2] = Cell(m, self.growthRate, self.devisionRate,
                                                 self.dt,self.t,self.T)
    def calculateDistanceToEdge(self,ix,iy):
        distLeft  = 0
        distRigth = 0
        distUp    = 0
        distDown  = 0
        
        i = ix
        #odleglosc do prawej
        while i+1 < self.CellMatrixXdim and self.CellMatrix[i+1,iy] is not None:
            distRigth += 1
            i += 1
        i = ix
        #odleglosc do lewej
        while i-1 >= 0 and self.CellMatrix[i-1,iy] is not None:
            distLeft += 1
            i -= 1
        j = iy
        #odleglosc do gory
        while j-1 >= 0 and self.CellMatrix[ix,j-1] is not None:
            distUp += 1
            j -= 1
        j = iy
        #odleglosc do dolu
        while j+1 < self.CellMatrixYdim and self.CellMatrix[ix,j+1] is not None:
            distDown += 1
            j += 1

        return distLeft,distRigth,distUp,distDown
        
    def moveCellsLeft(self,ix,iy):
        for i in range(1,ix+1):
            self.CellMatrix[i-1,iy] = self.CellMatrix[i,iy]
        self.CellMatrix[ix,iy] = None
    def moveCellsRight(self,ix,iy):
        for i in range(self.CellMatrixXdim-1,ix,-1):
            self.CellMatrix[i,iy] = self.CellMatrix[i-1,iy]
        self.CellMatrix[ix,iy] = None
    def moveCellsUp(self,ix,iy):
        for i in range(1,iy+1):
            self.CellMatrix[ix,i-1] = self.CellMatrix[ix,i]
        self.CellMatrix[ix,iy] = None
    def moveCellsDown(self,ix,iy):
        for i in range(self.CellMatrixYdim-1,iy,-1):
            self.CellMatrix[ix,i] = self.CellMatrix[ix,i-1]
        self.CellMatrix[ix,iy] = None
    def addCell(self, ix, iy, m):
        if self.CellMatrix[ix,iy] == None:
            self.CellMatrix[ix,iy] = Cell(m, self.growthRate, self.devisionRate,
                                                 self.dt,self.t,self.T)
        else:
            dists = self.calculateDistanceToEdge(ix,iy)
            if dists[0] == min(dists):
                self.moveCellsLeft(ix,iy)
            elif dists[1] == min(dists):
                self.moveCellsRight(ix,iy)
            elif dists[2] == min(dists):
                self.moveCellsUp(ix,iy)
            elif dists[3] == min(dists):
                self.moveCellsDown(ix,iy)
            self.CellMatrix[ix,iy] = Cell(m, self.growthRate, self.devisionRate,
                                                 self.dt,self.t,self.T)
    def updateCells(self):
        for ix in range(self.CellMatrixXdim):
            for iy in range(self.CellMatrixYdim):
                if self.CellMatrix[ix,iy] is not None:
                    self.CellMatrix[ix,iy].update(self.dt)
    def devideCells(self):
        for ix in range(self.CellMatrixXdim):
            for iy in range(self.CellMatrixYdim):
                if self.CellMatrix[ix,iy] is not None and self.CellMatrix[ix,iy].markedForDevision == 1:
                    newM = self.CellMatrix[ix,iy].m/2
                    self.CellMatrix[ix,iy].CellAfterDevision(self.growthRate, self.devisionRate,
                                                             self.dt, self.t, self.T)
                    self.addCell(ix,iy,newM)
    
    def setMaxRadius(self, i, singlePoint = False, maxRadius = 10):
        radius = 0
        if singlePoint == True:
            for cell in self.solution[i]:
                radius = max(radius,np.abs(cell[0]))
        elif singlePoint == False:    
            for it in range(min(i,len(self.solution))):
                for cell in self.solution[it]:
                    radius = max(radius,np.abs(cell[0]))
        else:
            for it in range(len(self.solution)):
                for cell in self.solution[it]:
                    radius = max(radius,np.abs(cell[0]))            
        self.maxRadius = min(radius, maxRadius)
        
    def setDisplaySize(self, i, singlePoint = False, maxRadius = 10):
        size = 0

        if singlePoint == True:
            for cell in self.solutionDisplay[i]:
                size = max(size,
                          np.abs(cell[0]-min(cell[2],maxRadius)),np.abs(cell[1]-min(cell[2],maxRadius)),
                          np.abs(cell[0]+min(cell[2],maxRadius)),np.abs(cell[1]+min(cell[2],maxRadius)))
        elif singlePoint == False:
            for it in range(min(i,len(self.solutionDisplay))):
                for cell in self.solutionDisplay[it]:
                    size = max(size,
                              np.abs(cell[0]-min(cell[2],maxRadius)),np.abs(cell[1]-min(cell[2],maxRadius)),
                              np.abs(cell[0]+min(cell[2],maxRadius)),np.abs(cell[1]+min(cell[2],maxRadius)))
        else:
            for it in range(len(self.solutionDisplay)):
                for cell in self.solutionDisplay[it]:
                    size = max(size,
                              np.abs(cell[0]-min(cell[2],maxRadius)),np.abs(cell[1]-min(cell[2],maxRadius)),
                              np.abs(cell[0]+min(cell[2],maxRadius)),np.abs(cell[1]+min(cell[2],maxRadius)))
        self.displaySize = size
        
    def updateCoordinates(self, i, singlePoint = False, maxRadius = 10):
        self.solutionDisplay = [0]*(i+1)
        
        if singlePoint == True:
            data = []
            for cell in self.solution[i]:
                data.append((
                    (cell[1] - self.CellMatrixXdim//2)*self.maxRadius*2,
                    (cell[2] - self.CellMatrixYdim//2)*self.maxRadius*2,
                     min(cell[0],maxRadius),
                ))
            self.solutionDisplay[i] = data
        elif singlePoint == False:    
            for it in range(min(i,len(self.solution))):
                data = []
                for cell in self.solution[it]:
                    data.append((
                        (cell[1] - self.CellMatrixXdim//2)*self.maxRadius*2,
                        (cell[2] - self.CellMatrixYdim//2)*self.maxRadius*2,
                         min(cell[0],maxRadius),
                    ))
                self.solutionDisplay[it] = data
        else:
            for it in range(len(self.solution)):
                data = []
                for cell in self.solution[it]:
                    data.append((
                        (cell[1] - self.CellMatrixXdim//2)*self.maxRadius*2,
                        (cell[2] - self.CellMatrixYdim//2)*self.maxRadius*2,
                         min(cell[0],maxRadius),
                    ))
                self.solutionDisplay[it] = data
            
    def growCells(self):
        self.solution = []
        self.solutionTimes = np.zeros((round(self.T/self.dt)+2))
        it = 0
        self.solution.append(self.getCellsData())
        self.solutionTimes[it] = self.t
        while self.t + self.dt < self.T:
            it += 1
            self.t += self.dt
            self.updateCells()
            self.devideCells()
            self.solution.append(self.getCellsData())
            self.solutionTimes[it] = self.t
            print('\rCzas {:>8.3f} .....Dodano'.format(self.t), end='', flush=True)
        
        print("\nUkonczono")

    def getCellsData(self):
        data = []
        for ix in range(self.CellMatrixXdim):
            for iy in range(self.CellMatrixYdim):
                if self.CellMatrix[ix,iy] is not None:
                    data.append((
                        self.CellMatrix[ix,iy].m,
                        ix,
                        iy
                    ))
        return data
    
    def getMasses(self, i):
        masses = []
        for cell in self.solution[i]:
            masses.append(cell[0])
        return masses

    def saveGrowthAnim(self, nazwa, interval, tMax, maxRadius = 10):
        
        # prep for animation
        
        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=2*1800)

        if tMax is not None:
            jj = np.argmin(np.abs(self.solutionTimes - tMax))
            frames = min(len(self.solution), jj) 
        else:
            frames = len(self.solution)
        
        self.setMaxRadius(frames, singlePoint = False, maxRadius = maxRadius)
        self.updateCoordinates(frames, singlePoint = False, maxRadius = maxRadius)
        self.setDisplaySize(frames, singlePoint = False, maxRadius = maxRadius)
        
        fig, ax = plt.subplots()

        ax.set_xlim((-self.displaySize, self.displaySize))
        ax.set_ylim((-self.displaySize, self.displaySize))
        
        tx = ax.set_title('Komorki t = 0')

        def animate(i):
            print('\rCzas {:>8.3f} .....Dodano'.format(i*self.dt), end='', flush=True)
                
            ax.artists.clear()
            for cell in self.solutionDisplay[i]:
                circle = plt.Circle((cell[0], cell[1]), cell[2], color='green', fill=False)
                ax.add_artist(circle)
            tx.set_text('Komorki t = {0}'.format(round(self.solutionTimes[i],2)))

        ani = animation.FuncAnimation(fig, animate, frames, interval=interval)

        ani.save(nazwa, writer=writer)

        plt.cla()
        
        print("\nZapisano")
        
    def displayCells(self, t=None, nazwa = None, maxRadius = 10):
        
        i = -1
        if t is not None:
            i = np.argmin(np.abs(self.solutionTimes - t))
        
        self.setMaxRadius(i, singlePoint = True, maxRadius = maxRadius)
        self.updateCoordinates(i, singlePoint = True, maxRadius = maxRadius)
        self.setDisplaySize(i, singlePoint = True, maxRadius = maxRadius)
        
        fig, ax = plt.subplots()

        tx = ax.set_title('Komorki t = ' + str(round(self.solutionTimes[i],2)))

        ax.set_xlim((-self.displaySize, self.displaySize))
        ax.set_ylim((-self.displaySize, self.displaySize))

        for cell in self.solutionDisplay[i]:
            circle = plt.Circle((cell[0], cell[1]), cell[2], color='green', fill=False)
            ax.add_artist(circle)

        if nazwa is not None:
            fig.savefig(nazwa)

        plt.show()
        
    def displayDistribution(self, nazwa, t, options):
        
        i = -1
        if t is not None:
            i = np.argmin(np.abs(self.solutionTimes - t))
        
        n, bins, patches = plt.hist(self.getMasses(i), bins = options['numBins'], normed=1, facecolor='green', alpha=0.75)

        plt.xlabel('Mass')
        plt.ylabel('Probability density')
        plt.title('Rozklad mas komorek dla czasu t = ' + str(round(self.solutionTimes[i],3)))
        if options['setCustomAxis'] == True:
            if options['xLeft'] is not None and options['xRight'] is not None:
                plt.xlim((options['xLeft'], options['xRight']))
            if options['yUp'] is not None and options['yDown'] is not None:
                plt.ylim((options['yDown'], options['yUp']))
        plt.grid(True)

        if nazwa is not None:
            plt.savefig(nazwa)
        
        plt.show()

    def saveGrowhHistAnim(self, nazwa, interval, options):

        Writer = animation.writers['ffmpeg']
        writer = Writer(fps=5, metadata=dict(artist='Me'), bitrate=2*1800)

        if options['tMax'] is not None:
            jj = np.argmin(np.abs(self.solutionTimes - options['tMax']))
            frames = min(len(self.solution), jj) 
        else:
            frames = len(self.solution)

        fig, ax = plt.subplots()

        plt.xlabel('Mass')
        plt.ylabel('Probability desity')
        plt.title('Rozklad mas komorek dla czasu t = ' + str(round(self.solutionTimes[0],3)))
        if options['setCustomAxis'] == True:   
            plt.axis([options['xLeft'], options['xRight'], options['yDown'], options['yUp']])
        plt.grid(True)
        
        tx = ax.set_title('Rozklad mas komorek dla czasu t = 0')
      
        def animate(i):
            print('\rCzas {:>8.3f} .....Dodano'.format(round(self.solutionTimes[i],3)), end='', flush=True)
                
            plt.cla()

            plt.xlabel('Masa')
            plt.ylabel('Probability desity')
            plt.title('Rozklad mas komorek dla czasu t = ' + str(round(self.solutionTimes[i],3)))
            if options['setCustomAxis'] == True:
                if options['xLeft'] is not None and options['xRight'] is not None:
                    plt.xlim((options['xLeft'], options['xRight']))
                if options['yUp'] is not None and options['yDown'] is not None:
                    plt.ylim((options['yDown'], options['yUp']))
            plt.grid(True)
            
            n, bins, patches = plt.hist(self.getMasses(i), normed=1, bins = options['numBins'], facecolor='green', alpha=0.75)
            tx.set_text('Rozklad mas komorek dla czasu t = {0}'.format(round(self.solutionTimes[i],3)))

        ani = animation.FuncAnimation(fig, animate, frames, interval=interval)

        ani.save(nazwa, writer=writer)
        
        plt.cla()

        print("\nZapisano")