
import time
import random

import smach
import smach_ros
import rospy

from functions import check_all_nodes, run_segmentation, move, moveManipulatorToHome, generate_grasps, save_object, changePosition
import config as cfg





class InitializeAll(smach.State):
    def __init__(self, outcomes=['All initialized', 'Not all initialized'],
                 ):
        smach.State.__init__(self, outcomes=outcomes)
        self.outcomes = outcomes

    def execute(self, userdata):

        ret = check_all_nodes()

        return 'All initialized' if ret else 'Not all initialized'


class WaitForCommand(smach.State):
    def __init__(self, outcomes=['Start working'],
                 ):
        smach.State.__init__(self, outcomes=outcomes)
        self.outcomes = outcomes

    def execute(self, userdata):

        # command = input('type ENTER to start')
        return 'Start working'


class RecognizeObjects(smach.State):
    def __init__(self, outcomes=['Object found', 'Object found with low confidence', 'No objects found'],
                 output_keys=['depth_masked']):
        smach.State.__init__(self, outcomes=outcomes,
                             output_keys=output_keys)

    def execute(self, userdata):

        ret = run_segmentation()
        
        if ret is None:
            return 'No objects found'
        else:
            # print(ret)
            nearest_depth_mask, cl, confidence, dist = ret

            print(cl, confidence)
            userdata.depth_masked = nearest_depth_mask

        if confidence > cfg.conf_thresh and dist < cfg.dist_thresh and cl != '':
            return 'Object found'
        else:
            return 'Object found with low confidence'


class GenerateGrasps(smach.State):
    def __init__(self, outcomes=['Grasps generated', 'Grasps not generated'],
                 input_keys=['depth_masked'],
                 output_keys=['grasping_poses']):
        smach.State.__init__(self, outcomes=outcomes,
                             input_keys=input_keys,
                             output_keys=output_keys)

    def execute(self, userdata):
        grasps = generate_grasps(userdata.depth_masked)

        userdata.grasping_poses = grasps

        if grasps is not None:
            return 'Grasps generated'
        else:
            return 'Grasps not generated'


class PlanAndMove(smach.State):
    def __init__(self, outcomes=['Planning failed', 'Moving sucessful'],
                 input_keys=['grasping_poses']):
        smach.State.__init__(self, outcomes=outcomes, input_keys=input_keys)

    def execute(self, userdata):
        return_value = move(userdata.grasping_poses)
        return return_value


class ReturnToInitState(smach.State):
    def __init__(self, outcomes=['Executed', 'Not executed']):
        smach.State.__init__(self, outcomes=outcomes)

    def execute(self, userdata):

        return_value = moveManipulatorToHome()
        if return_value is None:
            return 'Executed'
        else:
            return 'Not executed'




class ChangePosition(smach.State):
    def __init__(self, outcomes=['Position changed', 'Too many attempts', 'Moving failed'],
                 input_keys=['depth_masked']):
        smach.State.__init__(self, outcomes=outcomes, input_keys=input_keys)
        self.counter = 0

    def execute(self, userdata):

        if self.counter > cfg.max_retry:
            return 'Too many attempts'
        self.counter += 1
        rospy.logwarn(f'Attempt {self.counter}')

        ret_val = changePosition(userdata.depth_masked)

        if ret_val is not None:
            return 'Moving failed'
        else:
            return 'Position changed'


class LearnNewObject(smach.State):
    def __init__(self, outcomes=['Object saved', 'Object not saved'],
                 input_keys=['depth_masked']):
        smach.State.__init__(self, outcomes=outcomes, input_keys=input_keys)

    def execute(self, userdata):

        ret = save_object(userdata.depth_masked)
        return 'Object saved' if ret else 'Object not saved'


sm = smach.StateMachine(outcomes=['Stopped'])


with sm:
    smach.StateMachine.add('INITIALIZATION', InitializeAll(),
                           transitions={
        'All initialized': 'WAIT FOR COMMAND',
        'Not all initialized': 'Stopped'
    })
    smach.StateMachine.add('WAIT FOR COMMAND', WaitForCommand(),
                           transitions={
        'Start working': 'RECOGNIZE OBJECTS'
    })

    smach.StateMachine.add('RECOGNIZE OBJECTS', RecognizeObjects(),
                           transitions={
        'Object found': 'GENERATE GRASPS',
        'Object found with low confidence': 'LEARN OBJECT',
        'No objects found': 'CHANGE POSITION',
    })

    smach.StateMachine.add('GENERATE GRASPS', GenerateGrasps(),
                           transitions={
        'Grasps generated': 'PLAN AND MOVE',
        'Grasps not generated': 'CHANGE POSITION',
        # 'Objects found with low confidence': 'CHANGE POSITION'
    })

    smach.StateMachine.add('PLAN AND MOVE', PlanAndMove(),
                           transitions={
        'Planning failed': 'CHANGE POSITION',
        'Moving sucessful': 'GO TO INITIAL STATE'
    })

    smach.StateMachine.add('GO TO INITIAL STATE', ReturnToInitState(),
                           transitions={
        'Executed': 'WAIT FOR COMMAND',
        'Not executed': 'Stopped'
    })

    smach.StateMachine.add('CHANGE POSITION', ChangePosition(),
                           transitions={
        'Position changed': 'RECOGNIZE OBJECTS',
        'Too many attempts': 'GO TO INITIAL STATE',
        'Moving failed': 'Stopped'
    })

    smach.StateMachine.add('LEARN OBJECT', LearnNewObject(),
                           transitions={
        'Object saved': 'RECOGNIZE OBJECTS',
        'Object not saved': 'Stopped'
    })

rospy.init_node('blablabla')
# sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
# sis.start()

outcome = sm.execute()

# rospy.spin()

# sis.stop()
