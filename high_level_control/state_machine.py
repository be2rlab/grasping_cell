#!/usr/bin/env python3

import time

import smach
import smach_ros
import rospy

from ros_functions import check_all_nodes, run_object_recognition, \
    moveObjectToBox, moveManipulatorToHome, generate_grasps, learn_object, changePosition
from utils import AttemptCounter
import config as cfg

counter = AttemptCounter()

class InitializeAll(smach.State):
    def __init__(self, outcomes=['All initialized', 'Not all initialized'],
                 ):
        smach.State.__init__(self, outcomes=outcomes)
        self.outcomes = outcomes

    def execute(self, userdata):

        ret = check_all_nodes()

        ret = True

        if not ret:
            return 'Not all initialized'

        rospy.logwarn('Go to initial position')
        # return_value = moveManipulatorToHome()

        # return 'All initialized'
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
    def __init__(self, outcomes=['Object found', 'Object found with low confidence', 'No objects found', 'CV not available'],
                 output_keys=['mask', 'depth_masked', 'class_name']):
        smach.State.__init__(self, outcomes=outcomes,
                             output_keys=output_keys)

    def execute(self, userdata):

        time.sleep(2)
        # userdata.depth_masked, mask, cl, confidence, dist = run_object_recognition()
        result = run_object_recognition()

        if result is None:
            return 'CV not available'
        elif len(result.mask.data) == 0:
            return 'No objects found'
        elif result.class_conf > cfg.conf_thresh and result.class_dist < cfg.dist_thresh and result.class_name != '':
            print(
                f'Found object: {result.class_name} with confidence {result.class_conf:.2f}, distance: {result.class_dist:.2f}')
            ret_val = 'Object found'
        else:
            print(
                f'Found object of class {result.class_name}, confidence: {result.class_conf:.2f}, distance: {result.class_dist:.2f}')
            ret_val = 'Object found with low confidence'
            
        userdata.class_name = result.class_name
        userdata.mask = result.mask

        return ret_val




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
        return_value = moveObjectToBox(userdata.grasping_poses, userdata.class_name)
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
                 input_keys=['mask'],
                 output_keys=['mask']):
        smach.State.__init__(self, outcomes=outcomes,
                             input_keys=input_keys, output_keys=output_keys)

    def execute(self, userdata):

        ret = learn_object(userdata.mask)
        if ret == 'Object saved':
            counter.reset()
        return ret


if __name__ == '__main__':

    rospy.init_node('state_machine')

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
            'CV not available': 'INITIALIZATION'
        })

        smach.StateMachine.add('GENERATE GRASPS', GenerateGrasps(),
                               transitions={
            'Grasps generated': 'PLAN AND MOVE',
            'Grasps not generated': 'CHANGE POSITION',
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
            'Object not saved': 'GO TO INITIAL STATE',
            'Grasps not generated': 'CHANGE POSITION'
        })


    outcome = sm.execute()


