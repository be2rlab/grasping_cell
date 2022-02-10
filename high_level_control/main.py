#!/usr/bin/env python3

import time
import random
from utils import AttemptCounter

import smach
import smach_ros
import rospy

from functions import check_all_nodes, run_segmentation, move, moveManipulatorToHome, generate_grasps, save_object, changePosition
import config as cfg

counter = AttemptCounter()


class InitializeAll(smach.State):
    def __init__(self, outcomes=['All initialized', 'Not all initialized'],
                 ):
        smach.State.__init__(self, outcomes=outcomes)
        self.outcomes = outcomes

    def execute(self, userdata):

        ret = check_all_nodes()

        if not ret:
            return 'Not all initialized'

        rospy.logwarn('Go to initial position')
        # return_value = moveManipulatorToHome()

        # return 'All initialized' if ret and return_value else 'Not all initialized'
        return 'All initialized' if ret else 'Not all initialized'


class WaitForCommand(smach.State):
    def __init__(self, outcomes=['Start working'],
                 ):
        smach.State.__init__(self, outcomes=outcomes)
        self.outcomes = outcomes

    def execute(self, userdata):

        command = input('type ENTER to start')
        return 'Start working'


class RecognizeObjects(smach.State):
    def __init__(self, outcomes=['Object found', 'Object found with low confidence', 'No objects found'],
                 output_keys=['mask', 'depth_masked', 'class_name']):
        smach.State.__init__(self, outcomes=outcomes,
                             output_keys=output_keys)

    def execute(self, userdata):

        time.sleep(2)
        userdata.depth_masked, mask, cl, confidence, dist = run_segmentation()

        userdata.mask = mask
        userdata.class_name = cl

        if len(mask.data) == 0:
            return 'No objects found'
        # else:
            # userdata.depth_masked, userdata.mask, cl, confidence, dist = ret
        elif confidence > cfg.conf_thresh and dist < cfg.dist_thresh and cl != '':
            print(
                f'Found object: {cl} with confidence {confidence:.2f}, distance: {dist:.2f}')
            return 'Object found'
        else:
            print(
                f'Found object of class {cl}, confidence: {confidence:.2f}, distance: {dist:.2f}')
            return 'Object found with low confidence'


class GenerateGrasps(smach.State):
    def __init__(self, outcomes=['Grasps generated', 'Grasps not generated'],
                 input_keys=['mask', 'class_name'],
                 output_keys=['grasping_poses', 'class_name']):
        smach.State.__init__(self, outcomes=outcomes,
                             input_keys=input_keys,
                             output_keys=output_keys)

    def execute(self, userdata):
        grasps = generate_grasps(userdata.mask)

        userdata.grasping_poses = grasps

        if grasps != None:
            return 'Grasps generated'
        else:
            return 'Grasps not generated'


class PlanAndMove(smach.State):
    def __init__(self, outcomes=['Planning failed', 'Moving sucessful'],
                 input_keys=['grasping_poses', 'class_name']):
        smach.State.__init__(self, outcomes=outcomes, input_keys=input_keys)

    def execute(self, userdata):
        return_value = move(userdata)
        if return_value == 'Moving successful':
            counter.reset()
        return return_value


class ReturnToInitState(smach.State):
    def __init__(self, outcomes=['Executed', 'Not executed']):
        smach.State.__init__(self, outcomes=outcomes)

    def execute(self, userdata):
        counter.reset()

        return_value = moveManipulatorToHome()
        return 'Executed' if return_value else 'Not executed'


class ChangePosition(smach.State):
    def __init__(self, outcomes=['Position changed', 'Too many attempts', 'Moving failed'],
                 input_keys=['mask', 'depth_masked']):
        smach.State.__init__(self, outcomes=outcomes, input_keys=input_keys)

    def execute(self, userdata):

        if counter.val > cfg.max_retry:
            return 'Too many attempts'

        ret_val = changePosition(userdata.depth_masked, counter.val)
        counter.update()
        rospy.logwarn(f'Attempt {counter.val}')
        return ret_val


class LearnNewObject(smach.State):
    def __init__(self, outcomes=['Object saved', 'Object not saved', 'Grasps not generated'],
                 input_keys=['depth_masked'],
                 output_keys=['depth_masked']):
        smach.State.__init__(self, outcomes=outcomes,
                             input_keys=input_keys, output_keys=output_keys)

    def execute(self, userdata):

        ret = save_object(userdata.depth_masked)
        if ret == 'Object saved':
            counter.reset()
        return ret


if __name__ == '__main__':

    rospy.init_node('blablabla')

    sm = smach.StateMachine(outcomes=['Stopped'])

    sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    sis.start()

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
        })

        smach.StateMachine.add('PLAN AND MOVE', PlanAndMove(),
                               transitions={
            'Planning failed': 'CHANGE POSITION',
            'Moving sucessful': 'RECOGNIZE OBJECTS',
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
            # 'Moving failed': 'Stopped'
            'Moving failed': 'RECOGNIZE OBJECTS'
        })

        smach.StateMachine.add('LEARN OBJECT', LearnNewObject(),
                               transitions={
            'Object saved': 'RECOGNIZE OBJECTS',
            'Object not saved': 'GO TO INITIAL STATE',
            'Grasps not generated': 'CHANGE POSITION'
        })

    # sis = smach_ros.IntrospectionServer('server_name', sm, '/SM_ROOT')
    # sis.start()

    outcome = sm.execute()

    # rospy.spin()

    # sis.stop()
