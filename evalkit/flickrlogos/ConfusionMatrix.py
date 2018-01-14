'''
Confusion matrix.

Created on 06.09.2010

@author: Stefan Romberg, stefan.romberg@informatik.uni-augsburg.de
'''
__version__ = "0.1"
__author__  = "Stefan Romberg"
__all__     = ['ConfusionMatrix']

import numpy as np
import matplotlib.pyplot as plt
#import pylab as P

#===============================================================================
# 
#===============================================================================

class ListOfListMat(list):
    """A simple matrix implemented as list of lists."""

    __slots__ = {}
    def __init__(self, rows, cols):
        list.__init__(self)

        for i in range(0, rows):
            self.append( [0]*cols )

class ConfusionMatrix(object):
    """Confusion Matrix."""

    def __init__(self):
        """Creates a new empty confusion matrix."""
        self.mat  = dict()

    def add_result(self, actual_class, detected_class, count=1):
        """Adds one classification results to the confusion matrix."""
        if actual_class in self.mat:
            row = self.mat[actual_class]

            if detected_class in row:
                row[detected_class] += count
            else:
                row[detected_class] = count

        else:
            row = dict()
            row[detected_class] = count
            self.mat[actual_class] = row

    def _keys(self):
        """Returns a list of column/row names in this confusion matrix."""
        key_set = set()
        for actual_class,detected_classes in self.mat.items():
            key_set.add(actual_class)
            for detected_class in detected_classes.keys():
                key_set.add(detected_class)

        key_set = sorted(list(key_set))

        # no-logo class should come first
        if "no-logo" in key_set:
            key_set.remove( "no-logo" )
            tmp = ["no-logo"]
            tmp.extend(key_set)
            key_set = tmp

        return list(key_set)

    def aslistoflist(self):
        """Returns the confusion matrix as list-of-lists."""
        class_names = self._keys()
        n           = len(class_names)

        idx_map = dict()
        for i,k in enumerate(class_names):
            idx_map[k] = i

        m = ListOfListMat(n, n)

        for actual_class,detected_classes in self.mat.items():
            for detected_class,count in detected_classes.items():
                m[ idx_map[actual_class] ][ idx_map[detected_class] ] = count

        return m

    def plot(self, title=None):
        """Plots the confusion matrix."""
#        conf_arr = [[33,2,0,0,0,0,0,0,0,1,3], [3,31,0,0,0,0,0,0,0,0,0], [0,4,41,0,0,0,0,0,0,0,1], [0,1,0,30,0,6,0,0,0,0,1], [0,0,0,0,38,10,0,0,0,0,0], [0,0,0,3,1,39,0,0,0,0,4], [0,2,2,0,4,1,31,0,0,0,2], [0,1,0,0,0,0,0,36,0,2,0], [0,0,0,0,0,0,1,5,37,5,1], [3,0,0,0,0,0,0,0,0,39,0], [0,0,0,0,0,0,0,0,0,0,38] ]
        conf_arr    = self.aslistoflist()
        class_names = self._keys()
        class_nums  = range(0, len(class_names))

        norm_conf = []
        for row_i in conf_arr:
            row_sum = float(sum(row_i))
            if row_sum > 0:
                normalized_row1 = [ float(j)/row_sum for j in row_i ]
                normalized_row = []
                for j in normalized_row1:
                    if j > 0:
                        normalized_row.append( max(j, 0.03) )
                    else:
                        normalized_row.append( 0 )
            else:
                normalized_row = [ 0 for j in row_i ]
            norm_conf.append(normalized_row)        
        
        #plt.clf()
        fig = plt.figure(figsize=(10,8))
        ax  = fig.add_subplot(111)
        
        mat = np.array(norm_conf)

        # flip upside-down
        mat           = np.flipud(mat)
        class_names_r = list(reversed(class_names))
        conf_arr      = reversed(conf_arr)
        
        res = ax.imshow(mat, cmap=plt.cm.get_cmap("jet"), interpolation='nearest')
        cb  = plt.colorbar(res, shrink=0.74, #pad=0.3,
                           #extend="both",
                           spacing="proportional",
                           orientation='vertical') #panchor=(1.0, 0.3))
                           
        dark_color   = (0,0,0)
        bright_color = (0.8,0.8,0.8)
        for i, cas in enumerate(conf_arr):
            for j, c in enumerate(cas):
                #p = norm_conf[i][j]
                p = mat[i,j]
                fcolor = dark_color
                if p < 0.35:
                    fcolor = bright_color
                if c == 0:
                    pass
                elif c >= 1 and c <= 9:
                    plt.text(j-.20, i-.25, c, fontsize=6, color=fcolor)
                elif c >= 10 and c <= 99:
                    plt.text(j-.40, i-.25, c, fontsize=6, color=fcolor)
                elif c >= 100 and c <= 999:
                    plt.text(j-.40, i-.25, c, fontsize=4, color=fcolor)
                elif c >= 1000 and c <= 9999:
                    plt.text(j-.40, i-.25, c, fontsize=3, color=fcolor)
                else:
                    raise Exception("Too big number: "+str(c))


        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("classified as")
        ax.set_ylabel("actual class")
        ax.set_xlim(-0.5, max(class_nums) + 0.5)
        ax.set_ylim(-0.5, max(class_nums) + 0.5)

        import matplotlib.lines as mpllines
        lines = ax.get_xticklines()
        for line in lines:
            line.set_marker(mpllines.TICKDOWN)
        lines = ax.get_yticklines()
        for line in lines:
            line.set_marker(mpllines.TICKLEFT)

        plt.yticks( class_nums, [ label+"  " for label in class_names_r ],
                    rotation=0, horizontalalignment='right',  fontsize=8)
        plt.xticks( class_nums, [ label+"  " for label in class_names ],
                    rotation=90, fontsize=8)

        plt.minorticks_off()
        plt.tick_params(direction="out", left=True, bottom=True, top=False, right=False)
        
        #plt.subplots_adjust(top=0.95, bottom=0.12, left=0.28, right=0.77)
        # note: 
        # needed to re-adjust shrink,pad value as matplotlib changed its behaviour
        plt.subplots_adjust(top=0.95, bottom=0.12, left=0.28, right=0.90)
        return plt

    def __len__(self):
        """Returns the size of the confusion matrix."""
        return len(self.mat)

if __name__ == '__main__':
    #===========================================================================
    # TESTS
    #===========================================================================
    # run all doctests
    import doctest
    doctest.testmod()
    
    cm = ConfusionMatrix()

    cm.add_result("cocacola", "puma")
    cm.add_result("cocacola", "puma")
    cm.add_result("cocacola", "puma")
    cm.add_result("cocacola", "puma")
    cm.add_result("cocacola", "cocacola")
    cm.add_result("cocacola", "nike")

    cm.add_result("puma", "cocacola")
    cm.add_result("puma", "puma")
    cm.add_result("puma", "puma")
    cm.add_result("puma", "puma")
    cm.add_result("puma", "puma")
    cm.add_result("puma", "puma")

    cm.add_result("dhl", "nike")
    cm.add_result("dhl", "dhl")
    cm.add_result("dhl", "dhl")
    cm.add_result("dhl", "dhl")

    cm.add_result("adidas", "adidas")
    cm.add_result("adidas", "adidas")
    cm.add_result("adidas", "adidas")
    cm.add_result("adidas", "adidas")
    cm.add_result("adidas", "adidas")
    cm.add_result("adidas", "adidas")
    cm.add_result("adidas", "adidas")
    cm.add_result("adidas", "nike")
    cm.add_result("adidas", "puma")

    plt = cm.plot()
    plt.show()

    print("All doctests passed.")
