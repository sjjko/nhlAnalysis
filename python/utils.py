__author__ = 'kos'
"""
class with small support routines
"""

class small_util_routines(object):
    """
    a class which does as its name says, little things
    """

    def __init__(self,VERBOSITY_LEVEL):
        """we simply set the constants we need, and initialize with a value set elsewhere"""
        self._GLOBAL_VERBOSITY_LEVEL=VERBOSITY_LEVEL

    def ToassertF(self,test,iscritical,*args):
        """tests for test and acts according to iscritical input flag

        Args:
            test: boolean to test for
            iscritical: program quits when set to one
            *args: a list of strings and numbers, concatenated to give the message displayed if test failed
        Returns:
            nothing
        """

        message=""
        for arg in args:
            message=message+str(arg)

        """simple routine for reporting errors - alternative to exception raising"""
        if not test:
            print message
            if iscritical:
                print "is a critical error - exit"
                exit(-1)

    def VerbosityF(self,verbosityLevel,*args):
        """simple output routine, which allows to set verbosity levels"""

        message=""
        for arg in args:
            message=message+str(arg)

        """print message according to the globally set verbosity level"""
        if self._GLOBAL_VERBOSITY_LEVEL>verbosityLevel:
                print message

    def EnumF(self,*sequential, **named):
        """generate and return an enumeration object - used to access integer indices using strings"""
        enums = dict(zip(sequential, range(len(sequential))), **named)
        return type('Enum', (), enums)
