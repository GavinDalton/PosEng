#!/usr/bin/env python3
"""
*PosEng* module for the WEAVE Positioner. (Previously FSTest)
This module contains a class definition for a fibre-based object that implements a set of 
engineering-level utility functions that were developed during the positioner integration. 

This module is meant to be imported and an instance of the class invoked for each fibre that 
is to be used, althoug some member functions have generic applications.

Usage example:

    >>> from PosEng import OneMove as om
    >>> from pos import *
    >>> a=om(35,POSLIB.PLATE_A,speed=30)
    >>> a.fibid
    35
    >>> a.plate
    PLATE_A
   
:author: gavin.dalton@physics.ox.ac.uk
"""

# Initialuse some standard stuff
import re
import pos
import numpy as np
import glob
import os
import POSLIB
from datetime import datetime
import time
from read_plate import ReadPlate
from read_field import ReadField
from move_fibre import MoveFibre
from movement import Movement
from position import Position,FullPosition
from pos import pos_config
from astropy.io import fits
from matplotlib import pyplot as plt

class OneMove:
    """
    The OneMove class definition

    In inital setup, a configuration file is read containing deployed positions for all tier 3 fibres 
    at close to maximum gate angles. This are stored as .XT, .YT and are used for moving the fibres off
    the porches so that the porch heights can be measures. If the fibre seleted is not tier 3, these will be    set to the park position.

    :param int fibid: The number of the fibre to use.
    :param plate: POSLIB ID of the plate we want to use (used for setting up file lookups).
    :parm int speed: Percentage of maximum available speed to use for local movements.

    """
    def __init__(self,fibid,plate=POSLIB.PLATE_A,speed=60):
        pos.fibre_position_store.online()
        self.gen_flag = False
        self.my_min_shift=1.
        self.plate=plate
        self.fibid=fibid
        self.fac = np.pi/180.
        self.speed=speed
        if (self.fibid > 1007):
            self.speed = 5
            print ("Forcing mIFU speed to 5%")
        self.start = pos.fibre_position_store.get_fibre(plate,fibid)
        rf=ReadField()
        self.ntarg = rf.read_field()
        rp=ReadPlate(self.plate)
        self.nfib = rp.read_plate()
#        assert (rp.enabled[fibid] == True)
        self.xpark = rp.xpark[fibid]
        self.ypark = rp.ypark[fibid]
        self.xpiv = rp.xpiv[fibid]
        self.ypiv = rp.ypiv[fibid]
        self.tier = rp.tier[fibid]
        self.max_length = rp.max_length[fibid]
        self.min_length = rp.min_length[fibid]
        self.zpark=self.parkz_read(1,fibid)
        dx = self.xpark-self.xpiv
        dy = self.ypark-self.ypiv
        th = np.arctan2(dy,dx)
        self.Tpark=(np.pi+th)/self.fac
        if (fibid in rf.fibre_list):
            self.XT = rf.targx[fibid]
            self.YT = rf.targy[fibid]
        else:
            self.XT=self.xpark
            self.YT=self.ypark
            print ('No target for this fibre.. using PARK position...!')
        self.set_zmaps()
        self.find_tlimits()
        self.move_to_park=False
        self.move_from_park=False
        self.expose=pos_config.get('fibre_measure.exposure_time.morta.mos',
                                    float)
        self.gexpose=pos_config.get('fibre_measure.exposure_time.morta.guide',
                                    float)
        self.focus = 'fibre'
        self.image_type=np.str(rp.mytype[fibid])
        print ("Image type is ",self.image_type)
        return

    def set_zmaps(self):
        """
        Internal function to define the plate flatness maps for the two robots as read from 
        the :file:`positioner.cfg` file. This function also picks up the various heights to be used 
        for offseting the flatness map for different operations.

        """
        nm=pos_config.get_dict(['z_axis','flatness',str(self.plate),'nona'],float)
        mm=pos_config.get_dict(['z_axis','flatness',str(self.plate),'morta'],float)
        self.plate_grab=pos_config.get('z_axis.plate_grab', float)
        self.image_fibre=pos_config.get('z_axis.image_fibre', float)
        self.plate_release=pos_config.get('z_axis.plate_release', float)
        self.image_prism_height=pos_config.get('z_axis.image_plate', float)
        self.clear_z=pos_config.get('gripper_limits.clear_z', float)
        nC=np.array([nm['z0'],nm['x'],nm['y'],nm['xy'],nm['x2'],nm['y2']],dtype=np.float)
        mC=np.array([mm['z0'],mm['x'],mm['y'],mm['xy'],mm['x2'],mm['y2']],dtype=np.float)
        self.zm=np.array([mC,nC])
        return

    def zvalue(self,x,y,robot):
        """
        Returns the value of the surface map at the requested coordinates for the requested robot.
        
        .. note::
           This function may need to change as we adopt different zmaps for elevation and rotation changes.

        :param: float x: The X-coordinate (robot coordinates, mm)
        :param: float y: The Y-coordinate (robot coordinates, mm)
        :param: int robot: Which robot's map to query. (0=Morta, 1=Nona)

        :return: {zval}. The value of the appropriate map at this location.
        :rtype: float
        """
        zval = np.dot(np.c_[np.ones(np.size(x)),x,y,x*y,x**2,y**2],self.zm[robot])
        return zval

    def parkz_read(self,robot,fib):
        """
        Returns the park height to use for the given robot, fibre combination. Heights are read from the 
        offset files defined in :file:`positioner.cfg`.

        .. note:: 

           This routine will have to change if the way in which the park heights are stored changes

        :param: int robot: Which robot to use.
        :param: int fib: Which fibre's park height to return

        :return: {parkz}. The park height to use.
        :rtype: float
        """
        
        if (robot == 0):
            self.offsetfile=pos_config.get_filename('morta.gripper_offset_map')
        else:
            self.offsetfile=pos_config.get_filename('nona.gripper_offset_map')
        if (self.plate == POSLIB.PLATE_A):
            myplate=0
        else:
            myplate=1
        self.zz=np.loadtxt(self.offsetfile,dtype=np.float64)
        return self.zz[fib*2+myplate,6]

    def parkz_tweak(self,robot):
        
        if (robot == 0):
            _offsetfile='morta_offset_map_rot0ZD15.txt'
            _hackedparksa='Porches_morta_zd15rot0.txt'
            _hackedparksb='Porches_mortaB_zd15rot0.txt'
            _newfile='morta_gripper_offset_map.txt_new'
        else:
            _offsetfile='nona_offset_map_rot0ZD15.txt'
            _hackedparksa='Porches_nona_zd15rot0.txt'
            _hackedparksb='Porches_nonaB_zd15rot0.txt'
            _newfile='nona_gripper_offset_map.txt_new'
        if (self.plate == POSLIB.PLATE_A):
            myplate=0
            _hackedparks = _hackedparksa
        else:
            myplate=1
            _hackedparks = _hackedparksb
        _zz=np.loadtxt(_offsetfile,dtype=np.float64)
        _new = np.loadtxt(_hackedparks, dtype=np.float64)
        for i in _new:
            _fib=int(i[0])
            _zz[_fib*2+myplate,6] = i[1]
            _fib=int(i[0])-1
            _zz[_fib*2+myplate,6] = i[1]+10
            _fib=int(i[0])-2
            _zz[_fib*2+myplate,6] = i[1]+20

        np.savetxt(_newfile,_zz,fmt='%d %d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')            
        return 

    def safe_pull_out(self,RR):
        """
        Returns a position that could be used to place the fibre out on the plate along the line between
        the fibre porch and the plate centre. 
        *NO Checking is done here as to whether this position is actually safe!*

        :param: float RR: radial distance from the plate centre for the target position.

        :return: {x,y}. Position corresponding to this fibre placed at the requested radius.
        :rtype: {float, float}.
        """
        _tt = np.pi+np.arctan2(-self.ypark,-self.xpark)
        xx = RR*np.cos(_tt)
        yy = RR*np.sin(_tt)
        return xx,yy

    def move_over_park_plate(self,robot):
        """
        Move the robot to a position above the porch of the fibre, as if to pick up.
        This routine will move from the current robot position UP to `clear_z` before traversing 
        to the fibre park position.
        """
        mf = MoveFibre(robot,self.xpark,self.ypark,self.range_angle(self.Tpark,robot),speed=self.speed)
        mf.image_type=self.image_type
        _xx,_yy,_zz,_tt,_flag=pos.plc_controller.get_robot_position(robot)
        _,_,_,_ = mf.carry_move(_xx,_yy,self.range_angle(_tt,robot),self.clear_z,take_image=False)
        mf.abort = False # Don't care about the imaging here
        if (robot == 0):
            _xp,_yp=pos.move_one_fibre_m.plate_to_robot(self.xpark,self.ypark)
        else:
            _xp,_yp=pos.move_one_fibre_n.plate_to_robot(self.xpark,self.ypark)
        _,_,_,_ = mf.carry_move(_xp,_yp,self.range_angle(self.Tpark,robot),self.clear_z,take_image=False)
        mf.abort = False # Don't care about the imaging here
        return

    def set_start_coords(self,robot):
        # assumption is that we have moved by hand to a position where we can see the fibre in park, so want to update the position_store with the true values.
        if (robot == 0):
            a,b,xx,yy = pos.move_one_fibre_m.measure(self.expose,self.image_type,self.focus)
        else:
            a,b,xx,yy = pos.move_one_fibre_n.measure(self.expose,self.image_type,self.focus)
        if (a == -1 or b == -1):
            print ("No fibre seen here.. suggest try ...measure() by hand?")
            return
        pos.fibre_position_store.park(self.plate,self.fibid,self.start.start_x,self.start.start_y,self.start.start_t*self.fac,self.xpark,self.ypark,self.start.start_t*self.fac)
        assert pos.fibre_position_store.state() == 'Fibres moving; '
        pos.fibre_position_store.end_move(self.plate, self.fibid,xx,yy,self.start.start_t*self.fac)
        self.start = pos.fibre_position_store.get_fibre(self.plate,self.fibid)
        return a,b,xx,yy

    def nudge(self,robot,dx,dy,dz=0,dt=0):
        """
        Move the chosen robot by offsets from its current position.

        :param: int robot: Which robot to move
        :param: float dx: Distance to move in X (mm)
        :param: float dy: Distance to move in Y (mm)
        :param: float dz: *Optional* Distance to move in Z (mm)
        :param: float dt: *Optional* Distance to move in theta (degrees)
        """
        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        xx2 = xx1+dx
        yy2 = yy1+dy
        zz2 = zz1+dz
        tt2 = tt1+dt
        fp2=FullPosition(xx2,yy2,tt2,zz2)
        _m = Movement(robot,fp2,self.speed)
        _m.move_all()
        _m.wait()
        return

    def find_tlimits(self):
        """
        Pick up the theta limits for the two robots from :file:`positioner.cfg` and store in a pair of arrays.
        """
        self.thigh=[]
        self.tlow=[]
        self.thigh.append(pos_config.get('morta.theta_axis.high_limit', float))
        self.tlow.append(pos_config.get('morta.theta_axis.low_limit', float))
        self.thigh.append(pos_config.get('nona.theta_axis.high_limit', float))
        self.tlow.append(pos_config.get('nona.theta_axis.low_limit', float))
        return

    def range_angle(self,angle,robot):
        """
        Map an angle into the range allowed for movement of the given robot using the limits as
        set from `find_tlimits`.

        :param: float angle: The desired angle (degrees).
        :param: int robot: Which robot to use.
        
        :return: {angle}. Useable representation of the angle.
        :rtype: float
        """
        if (angle > self.thigh[robot]):
            angle = angle - 360.
        if (angle < self.tlow[robot]):
            angle= angle + 360.
        return angle

    def last_image_name(self,robot):
        """
        Utility function that returns the filename of the last image taken by one of the cameras.

        :param: int robot: which robot to look for.

        :return:  {filename}. The full pathname of the image.
        :rtype: str
        """
        if (robot == 0):
            _sub = 'gripper_m'
        else:
            _sub = 'gripper_n'
        _test = '/home/pos_eng/WEAVE/data/cameras/'+_sub+'*'
        latest = max(glob.glob(_test),key=os.path.getctime)
        print(latest)
        return latest

    def update_image_header(self,name,robot,xp,yp):
        """
        Utility routine to add some information to the FITS header of a camera image for future reference.
        
        :param: str name: The filename of the image to update.
        :param: int robot: Which robot to gather information from.
        :param: float xp: X Pixel position to add to the header.
        :param: float yp: Y Pixel position to add to the header.

        """
        f=fits.open(name,mode='update')
        f[0].header['FIBRE'] = self.fibid
        f[0].header['PLATE'] = str(self.plate)
        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        f[0].header['XPIX'] = xp
        f[0].header['YPIX'] = yp
        f[0].header['XPLATE'] = xx1
        f[0].header['YPLATE'] = yy1
        f[0].header['TPLATE'] = zz1
        f[0].header['ZPLATE'] = tt1
        f.flush()
        f.close()
        return 0

    def move_over_fibre_plate(self,robot):
        """
        Move the robot to a position above the currently known position of the fibre on the plate, 
        as if to pick up.
        This routine will move from the current robot position UP to `clear_z` before traversing 
        to the fibre's position.

        :param: int robot: Which robot to use.
        """
        
        _current = pos.fibre_position_store.get_fibre(self.plate, self.fibid)
        _x = _current.start_x
        _y = _current.start_y
        dx = _x - self.xpiv
        dy = _y - self.ypiv
        th=np.arctan2(dy,dx)
        _t = self.range_angle((np.pi + th)/self.fac,robot)
        mf = MoveFibre(robot,_x,_y,_t,speed=self.speed)
        mf.image_type=self.image_type
        _xx,_yy,_zz,_tt,_flag=pos.plc_controller.get_robot_position(robot)
        mf.carry_move(_xx,_yy,self.range_angle(_tt,robot),self.clear_z,take_image=False)
        mf.abort = False
        if (robot == 0):
            _xp,_yp=pos.move_one_fibre_m.plate_to_robot(_x,_y)
        else:
            _xp,_yp=pos.move_one_fibre_n.plate_to_robot(_x,_y)
        mf.carry_move(_xp,_yp,_t,self.clear_z,take_image=False)
        mf.abort = False
        return

    def measure_t_coeff(self,robot,interval=5,logfile='tcoefflog.txt'):
        """
        Assume fibre is out on the plate. Move the robot over it and cycle z axis up and down 
        with repeated measurements of the pixel position of the fibre.

        :param: int robot: Which robot to use.
        :param: int interval: *optional* delay between moves.
        :param: str logfile: *optional* the filename to log the results... default is :file:`tcoefflog.txt`.

        """
        _this = pos.fibre_position_store.get_fibre(self.plate,self.fibid)
        if (_this.enabled == False):
            raise Exception ("Fibre is not enabled")
        if (_this.parked):
            raise Exception ("Fibre is parked... please move it somewhere on the plate and try again!")
        move_one_fibre_me = pos.move_one_fibre_m
        zz = self.zvalue(_this.start_x,_this.start_y,robot)[0]+self.plate_release
        tindz = 3
        tindy = 2
        if (robot == 1):
            move_one_fibre_me = pos.move_one_fibre_n
            tindz = 8
            tindy = 7
        self.move_over_fibre_plate(robot)
        startz = pos_config.get('gripper_limits.clear_z',float)
        self.nudge(robot,0.,0.,dz=-startz) # move up to z=0
        t0 = time.time()
        _out = np.zeros(1,dtype=[('robot','int'),('time','float'),('xp','float'),('yp','float'),('xplate','float'),('yplate','float'),('temp_z','float'),('temp_y','float')])
        _out['robot'] = robot 
        for j in range(0,1000):
            f=open(logfile,'ab')
            self.nudge(robot,0.,0.,dz=zz)
            time.sleep(interval)
            _out['xp'],_out['yp'],_out['xplate'],_out['yplate'] = move_one_fibre_me.measure(100,'mos','fibre')
            _out['time'] = (time.time()-t0)/60.
            _ta = pos.plc_controller.get_temperatures()
            _out['temp_z'] = _ta[tindz]
            _out['temp_y'] = _ta[tindy]
            np.savetxt(f,_out,fmt="%1d %5d %8.3f %8.3f %8.3f %8.3f %6.2f %6.2f")
            f.close()
            print( 'Going up ',_out['time'], j)
            __x,__y,__z,__t,__gr = pos.plc_controller.get_robot_position(robot)
            self.nudge(robot,0.,0.,dz=-__z)
            time.sleep(0.5)
        return
    

    def measure_scale_upper(self,robot,exp,spot,theta_offset=0,offx=33.946,offy=0.435):
        """
        Move the robot to image one of the fiducial marks on the plate, centre the spot on the camera 
        and them move it around the image to provide an estimate of the image scale in this configuration.

        :param: int robot: Which robot to use.
        :param: float exp: Exposure time to use for the images (ms), typically 50.
        :param: int spot: ID number of the spot to use (0..356)
        :param: float theta_offset: *optional* Offset from `theta_park` to use for this measurement.
        :param: float offx: *optional* offset in mm along the camera y-axis of the upper focus from the gripper focus
        :parm: float offy: *optional* offset in mm along the camera x-axis of the upper focus from the gripper focus

        :return: {_out}. Array of the pixel positions and corresponding reported plate positions of the spot.
        :rtype: Numpy 2d Array
        """

        # At the current position, move to a height suitable to focus on the top of the prism
# Need to move 1mm towards the pivot to make sure we see the button shoulder. DO NOT WANT TO DO THIS IN PARK...
        thmin = pos_config.get('morta.theta_axis.low_limit',float)
        thmax = pos_config.get('morta.theta_axis.high_limit',float)
        scalex=pos_config.get('cameras.gripper_m.plate.pixel_scale_x',float)
        scaley=pos_config.get('cameras.gripper_m.plate.pixel_scale_y',float)
        ax_x = pos_config.get('cameras.gripper_m.plate.rotation_axis.x',float)
        ax_y = pos_config.get('cameras.gripper_m.plate.rotation_axis.y',float)
        move_one_fibre_me = pos.move_one_fibre_m
        indz = 3
        if (robot == 1):
            move_one_fibre_me = pos.move_one_fibre_n
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
            scalex=pos_config.get('cameras.gripper_n.plate.pixel_scale_x',float)
            scaley=pos_config.get('cameras.gripper_n.plate.pixel_scale_y',float)
            ax_x = pos_config.get('cameras.gripper_n.plate.rotation_axis.x',float)
            ax_y = pos_config.get('cameras.gripper_n.plate.rotation_axis.y',float)
            indz=8
        _x = pos.plate_info(self.plate).fiducials[spot].x
        _y = pos.plate_info(self.plate).fiducials[spot].y
        zspot = pos_config.get('z_axis.image_plate',float)
        _z = self.zvalue(_x,_y,robot)[0]+zspot
        (_xx,_yy) = move_one_fibre_me.plate_to_robot(_x,_y)
        ff=FullPosition(_xx,_yy,-180.,_z)
        _m=Movement(robot,ff,self.speed)
        _m.move_all()
        _m.wait()
        _out = np.zeros(6,dtype=[('robot','int'),('xp','float'),('yp','float'),('xplate','float'),('yplate','float'),('temp_z','float'),('theta','float')])
        pos.plc_controller.gripper_spotlight_on(robot)
        # jaws shold now be over the spot
        _tt = np.pi-theta_offset*self.fac
        offsety = offx  # how far to move back down the vane to get a good image (mm)
        offsetx = offy
        offset = -ax_y*scaley
        basex = pos_config.get('plate_measure.camera_pixel.x',float)
        basey = pos_config.get('plate_measure.camera_pixel.y',float)
        print('Calculated offsets',((-ax_y+basey)*scaley,(ax_x-basex)*scalex))
        ax_yprime = offsety/scaley
        ax_xprime = offsetx/scalex
        print('Calculated axis_x,axis_y:',ax_xprime,ax_yprime)
        _dx2 = offsety*np.cos(_tt)-offsetx*np.sin(_tt)
        _dy2 = offsety*np.sin(_tt)+offsetx*np.cos(_tt)
        zview=_z
        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        _dz = zview-zz1
        _th2 = tt1-theta_offset
        if (_th2 < thmin):
            _th2 = _th2 + 360.
        if (_th2 > thmax):
            _th2 = _th2 - 360.
        dth = _th2 - tt1
        self.nudge(robot,_dx2,_dy2,dz=_dz,dt=dth)
        time.sleep(0.5)
        _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'fid','plate')
        dxp = (_xp-basex)*scaley
        dyp = (_yp-basey)*scalex
        print (dxp,dyp)
        self.nudge(robot,dyp,dxp)
        xx2,yy2,zz2,tt2,_flag=pos.plc_controller.get_robot_position(robot)
        for i in range(0,6):
            _out['robot'][i] = robot
            _out['theta'][i] = tt2
            _out['temp_z'][i] = pos.plc_controller.get_temperatures()[indz]
        time.sleep(0.5)
        _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'fid','plate')
#        if (input('Are we close enough to the centre?') == 'y'):
        if (True):
            # should now be on the centre of the chip
            _out['xp'][0] = _xp
            _out['yp'][0] = _yp
            _out['xplate'][0] = _xcalc
            _out['yplate'][0] = _ycalc
            k=0
            for (jx,jy) in [(1,1),(0,-2),(-2,0),(0,2),(1,-1)]:
                k = k + 1
                self.nudge(robot,jx,jy)
                time.sleep(0.5)
                _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'fid','plate')
                _out['xp'][k] = _xp
                _out['yp'][k] = _yp
                _out['xplate'][k] = _xcalc
                _out['yplate'][k] = _ycalc
        pos.plc_controller.gripper_spotlight_off(robot)
        print("Using scales from positioner.cfg:", scalex,scaley)
        print("X Positions: ",_out['xplate'].mean(),_out['xplate'].std())
        print("Y Positions: ",_out['yplate'].mean(),_out['yplate'].std())
        xscale1 = 2000./(_out['xp'][2]-_out['xp'][1])
        xscale2 = 2000./(_out['xp'][3]-_out['xp'][4])
        yscale1 = 2000./(_out['yp'][3]-_out['yp'][2])
        yscale2 = 2000./(_out['yp'][4]-_out['yp'][1])
        print('New Xscales ',xscale1,xscale2)
        print('New Yscales ',yscale1,yscale2)
        return _out

    def measure_spot_upper(self,robot,exp,spot,theta_offset=0,mydx=0,mydy=0,offx=33.946,offy=0.435):
# At the current position, move to a height suitable to focus on the top of the prism
# Need to move 1mm towards the pivot to make sure we see the button shoulder. DO NOT WANT TO DO THIS IN PARK...
        thmin = pos_config.get('morta.theta_axis.low_limit',float)
        thmax = pos_config.get('morta.theta_axis.high_limit',float)
        move_one_fibre_me = pos.move_one_fibre_m
        if (robot == 1):
            move_one_fibre_me = pos.move_one_fibre_n
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
        _current = pos.fibre_position_store.get_fibre(self.plate, self.fibid)
        _x = pos.plate_info(self.plate).fiducials[spot].x
        _y = pos.plate_info(self.plate).fiducials[spot].y
        zspot = pos_config.get('z_axis.image_plate',float)
        _z = self.zvalue(_x,_y,robot)[0]+zspot
        (_xx,_yy) = move_one_fibre_me.plate_to_robot(_x,_y)
        ff=FullPosition(_xx+mydx,_yy+mydy,-180.,_z)
        _m=Movement(robot,ff,self.speed)
        _m.move_all()
        _m.wait()
        pos.plc_controller.gripper_spotlight_on(robot)
        # jaws shold now be over the spot
        _tt = np.pi-theta_offset*self.fac
        offsety = offx  # how far to move back down the vane to get a good image (mm)
        offsetx = offy
        _dx2 = offsety*np.cos(_tt)-offsetx*np.sin(_tt)
        _dy2 = offsety*np.sin(_tt)+offsetx*np.cos(_tt)
        zview=_z
        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        _dz = zview-zz1
        _th2 = tt1-theta_offset
        if (_th2 < thmin):
            _th2 = _th2 + 360.
        if (_th2 > thmax):
            _th2 = _th2 - 360.
        dth = _th2 - tt1
        self.nudge(robot,_dx2,_dy2,dz=_dz,dt=dth)
        time.sleep(0.5)
        _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'fid','plate')
        _m=Movement(robot,ff,self.speed)
        _m.move_all()
        _m.wait()
        time.sleep(0.5)
        pos.plc_controller.gripper_spotlight_off(robot)
        return _xp,_yp,_xcalc,_ycalc

    def measure_fibre_upper2_test(self,robot,exp,spot,theta_offset=0,mydx=0,mydy=0,offx=33.394,offy=0.06):
# At the current position, move to a height suitable to focus on the top of the prism
# Need to move 1mm towards the pivot to make sure we see the button shoulder. DO NOT WANT TO DO THIS IN PARK...
        thmin = pos_config.get('morta.theta_axis.low_limit',float)
        thmax = pos_config.get('morta.theta_axis.high_limit',float)
        xpixscale = pos_config.get('cameras.gripper_m.plate.pixel_scale_x',float)
        ypixscale = pos_config.get('cameras.gripper_m.plate.pixel_scale_y',float)
        move_one_fibre_me = pos.move_one_fibre_m
        if (robot == 1):
            move_one_fibre_me = pos.move_one_fibre_n
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
            xpixscale = pos_config.get('cameras.gripper_n.plate.pixel_scale_x',float)
            ypixscale = pos_config.get('cameras.gripper_n.plate.pixel_scale_y',float)
        _current = pos.fibre_position_store.get_fibre(self.plate, self.fibid)
        _x = pos.plate_info(self.plate).fiducials[spot].x
        _y = pos.plate_info(self.plate).fiducials[spot].y
        zspot = pos_config.get('z_axis.image_plate',float)
#        (_xx,_yy) = move_one_fibre_me.plate_to_robot(_x,_y)
        _xx = -0.481
        _yy = -0.659
        _z = self.zvalue(_xx,_yy,robot)[0]+zspot
        # jaws shold now be over the spot
        _tt = np.pi-theta_offset*self.fac
        offsety = offx  # how far to move back down the vane to get a good image (mm)
        offsetx = offy
        _dx2 = offsety*np.cos(_tt)-offsetx*np.sin(_tt)
        _dy2 = offsety*np.sin(_tt)+offsetx*np.cos(_tt)
        zview=_z
#        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        tt1 = -180.
        zz1 = 1.65
        _dz = zview-zz1
        _th2 = tt1-theta_offset
        if (_th2 < thmin):
            _th2 = _th2 + 360.
        if (_th2 > thmax):
            _th2 = _th2 - 360.
        dth = _th2 - tt1
        (x1,y1)=move_one_fibre_me.robot_to_plate(_xx,_yy)
        (x2,y2)=move_one_fibre_me.plate_to_robot(x1+mydx+_dx2,y1+mydy+_dy2)
#        (x2,y2)=(_xx+mydx+_dx2,_yy+mydy+_dy2)
        ff=FullPosition(x2,y2,-180.+dth,_z+_dz)
        print(ff)
        _m=Movement(robot,ff,self.speed)
        _m.move_all()
        _m.wait()
        print(pos.plc_controller.get_robot_position(robot))
        _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'mos','plate')
# Now put this spot on 640,480 and return the robot coordinates:
        dxp = (640 - _xp)*xpixscale
        dyp = (480 - _yp)*ypixscale
        _dx3 = dyp*np.cos(_tt)-dxp*np.sin(_tt)
        _dy3 = dyp*np.sin(_tt)+dxp*np.cos(_tt)
        ff=FullPosition(x2+_dx3,y2+_dy3,-180.+dth,_z+_dz)
        print (ff)
        _m=Movement(robot,ff,self.speed)
        _m.move_all()
        _m.wait()
        _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'mos','plate')
        _xact,_yact,_zact,_tact,_=pos.plc_controller.get_robot_position(robot)
        return _xp,_yp,_xact,_yact

    def measure_spot_upper2_test(self,robot,exp,spot,theta_offset=0,mydx=0,mydy=0,offx=33.394,offy=0.06):
# At the current position, move to a height suitable to focus on the top of the prism
# Need to move 1mm towards the pivot to make sure we see the button shoulder. DO NOT WANT TO DO THIS IN PARK...
        thmin = pos_config.get('morta.theta_axis.low_limit',float)
        thmax = pos_config.get('morta.theta_axis.high_limit',float)
        move_one_fibre_me = pos.move_one_fibre_m
        if (robot == 1):
            move_one_fibre_me = pos.move_one_fibre_n
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
        _current = pos.fibre_position_store.get_fibre(self.plate, self.fibid)
        _x = pos.plate_info(self.plate).fiducials[spot].x
        _y = pos.plate_info(self.plate).fiducials[spot].y
        zspot = pos_config.get('z_axis.image_plate',float)
#        (_xx,_yy) = move_one_fibre_me.plate_to_robot(_x,_y)
        _xx = -40.323
        _yy = 99.706
        _z = self.zvalue(_x,_y,robot)[0]+zspot
        pos.plc_controller.gripper_spotlight_on(robot)
        # jaws shold now be over the spot
        _tt = np.pi-theta_offset*self.fac
        offsety = offx  # how far to move back down the vane to get a good image (mm)
        offsetx = offy
        _dx2 = offsety*np.cos(_tt)-offsetx*np.sin(_tt)
        _dy2 = offsety*np.sin(_tt)+offsetx*np.cos(_tt)
        zview=_z
#        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        tt1 = -180.
        zz1 = _z
        _dz = zview-zz1
        _th2 = tt1-theta_offset
        if (_th2 < thmin):
            _th2 = _th2 + 360.
        if (_th2 > thmax):
            _th2 = _th2 - 360.
        dth = _th2 - tt1
        (x1,y1)=move_one_fibre_me.robot_to_plate(_xx,_yy)
        (x2,y2)=move_one_fibre_me.plate_to_robot(x1+mydx+_dx2,y1+mydy+_dy2)
#        (x2,y2)=(_xx+mydx+_dx2,_yy+mydy+_dy2)
        ff=FullPosition(x2,y2,-180.+dth,_z+_dz)
        print(ff)
        _m=Movement(robot,ff,self.speed)
        _m.move_all()
        _m.wait()
        print(pos.plc_controller.get_robot_position(robot))
        _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'fid','plate')
        pos.plc_controller.gripper_spotlight_off(robot)
        return _xp,_yp,_xcalc,_ycalc

    def measure_spot_upper2(self,robot,exp,spot,theta_offset=0,mydx=0,mydy=0,offx=33.946,offy=0.435):
# At the current position, move to a height suitable to focus on the top of the prism
# Need to move 1mm towards the pivot to make sure we see the button shoulder. DO NOT WANT TO DO THIS IN PARK...
        thmin = pos_config.get('morta.theta_axis.low_limit',float)
        thmax = pos_config.get('morta.theta_axis.high_limit',float)
        move_one_fibre_me = pos.move_one_fibre_m
        if (robot == 1):
            move_one_fibre_me = pos.move_one_fibre_n
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
        _current = pos.fibre_position_store.get_fibre(self.plate, self.fibid)
        _x = pos.plate_info(self.plate).fiducials[spot].x
        _y = pos.plate_info(self.plate).fiducials[spot].y
        zspot = pos_config.get('z_axis.image_plate',float)
        (_xx,_yy) = move_one_fibre_me.plate_to_robot(_x,_y)
        _z = self.zvalue(_x,_y,robot)[0]+zspot
        pos.plc_controller.gripper_spotlight_on(robot)
        # jaws shold now be over the spot
        _tt = np.pi-theta_offset*self.fac
        offsety = offx  # how far to move back down the vane to get a good image (mm)
        offsetx = offy
        _dx2 = offsety*np.cos(_tt)-offsetx*np.sin(_tt)
        _dy2 = offsety*np.sin(_tt)+offsetx*np.cos(_tt)
        zview=_z
#        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        tt1 = -180.
        zz1 = _z
        _dz = zview-zz1
        _th2 = tt1-theta_offset
        if (_th2 < thmin):
            _th2 = _th2 + 360.
        if (_th2 > thmax):
            _th2 = _th2 - 360.
        dth = _th2 - tt1
        ff=FullPosition(_xx+mydx+_dx2,_yy+mydy+_dy2,-180.+dth,_z+_dz)
        _m=Movement(robot,ff,self.speed)
        _m.move_all()
        _m.wait()
        _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'fid','plate')
        pos.plc_controller.gripper_spotlight_off(robot)
        return _xp,_yp,_xcalc,_ycalc

    def measure_fibre_upper(self,robot,exp,theta_offset=0):
# At the current position, move to a height suitable to focus on the top of the prism
# Need to move 1mm towards the pivot to make sure we see the button shoulder. DO NOT WANT TO DO THIS IN PARK...
        thmin = pos_config.get('morta.theta_axis.low_limit',float)
        thmax = pos_config.get('morta.theta_axis.high_limit',float)
        move_one_fibre_me = pos.move_one_fibre_m
        if (robot == 1):
            move_one_fibre_me = pos.move_one_fibre_n
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
        _current = pos.fibre_position_store.get_fibre(self.plate, self.fibid)
        _x = _current.start_x
        _y = _current.start_y
        _dx1 = _x-self.xpiv
        _dy1 = _y-self.ypiv
        _tt = np.pi+np.arctan2(_dy1,_dx1)-theta_offset*self.fac
        offset = 33.5  # how far to move back down the vane to get a good image (mm)
        _dx2 = offset*np.cos(_tt)
        _dy2 = offset*np.sin(_tt)
        zview=self.image_prism_height+self.zvalue(_x,_y,robot)[0]
        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        _dz = zview-zz1
        _th2 = tt1-theta_offset
        if (_th2 < thmin):
            _th2 = _th2 + 360.
        if (_th2 > thmax):
            _th2 = _th2 - 360.
        dth = _th2 - tt1
        self.nudge(robot,_dx2,_dy2,dz=_dz-2.25,dt=dth)
        time.sleep(0.5)
        _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(exp,'mos','plate')
        self.nudge(robot,-_dx2,-_dy2,dz=-_dz+2.25,dt=-dth)
        time.sleep(0.5)
        return _xp,_yp,_xcalc,_ycalc

    def measure_prism(self,robot,exp):
# At the current position, move to a height suitable to focus on the top of the prism
# Need to move 1mm towards the pivot to make sure we see the button shoulder. DO NOT WANT TO DO THIS IN PARK...
        pos.plc_controller.gripper_spotlight_on(robot)
        _current = pos.fibre_position_store.get_fibre(self.plate, self.fibid)
        _x = _current.start_x
        _y = _current.start_y
        _dx1 = _x-self.xpiv
        _dy1 = _y-self.ypiv
        _tt = np.pi+np.arctan2(_dy1,_dx1)
        offset = 33.5  # how far to move back down the vane to get a good image (mm)
        _dx2 = offset*np.cos(_tt)
        _dy2 = offset*np.sin(_tt)
        zview=self.image_prism_height+self.zvalue(_x,_y,robot)[0]
        xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
        _dz = zview-zz1
        self.nudge(robot,_dx2,_dy2,dz=_dz)
        time.sleep(1.)
        if (robot == 0):
            _xp,_yp,_xcalc,_ycalc=pos.move_one_fibre_m.measure(exp,'mos','plate')
        else:
            _xp,_yp,_xcalc,_ycalc=pos.move_one_fibre_n.measure(exp,'mos','plate')
        pos.plc_controller.gripper_spotlight_off(robot)
        self.nudge(robot,0,0,dz=1.5)
        _name = self.last_image_name(robot)
        time.sleep(1.)
        if (robot == 0):
            _xp,_yp,_xcalc,_ycalc=pos.move_one_fibre_m.measure(exp,'mos','plate')
        else:
            _xp,_yp,_xcalc,_ycalc=pos.move_one_fibre_n.measure(exp,'mos','plate')
        self.update_image_header(_name,robot,_xp,_yp)
        return _name

    def find_grasp(self,robot,parkit=True):
        if (self.image_type == 'guide'):
            return self.find_grasp_guide(robot,parkit=parkit)
        self.zpark=self.parkz_read(robot,self.fibid)
        xx,yy=self.safe_pull_out(135)
        pos.unpark_fibre(robot,self.fibid,xx,yy)
        _x,_y,__z,_th,_flag=pos.plc_controller.get_robot_position(robot)
#        self.move_over_fibre_plate(robot)
        myp=Position(0.,0.,0.)
        mf=Movement(robot,myp)
        mf.move_z(self.zvalue(xx,yy,robot)[0]+self.plate_release)
        mf.wait()
        time.sleep(1.)
        mf.close_jaws()
        if (mf.wait()):
            if (robot == 1):
                self.expose=pos_config.get('fibre_measure.exposure_time.nona.mos',float)
                xrot=pos_config.get('cameras.gripper_n.fibre.rotation_axis.x',float)
                yrot=pos_config.get('cameras.gripper_n.fibre.rotation_axis.y',float)
                scale=pos_config.get('cameras.gripper_n.fibre.pixel_scale',float)
                xpix,ypix,xplate,yplate=pos.move_one_fibre_n.measure(self.expose,'mos','fib')
            else:
                self.expose=pos_config.get('fibre_measure.exposure_time.morta.mos',float)
                xrot=pos_config.get('cameras.gripper_m.fibre.rotation_axis.x',float)
                yrot=pos_config.get('cameras.gripper_m.fibre.rotation_axis.y',float)
                scale=pos_config.get('cameras.gripper_m.fibre.pixel_scale',float)
                xpix,ypix,xplate,yplate=pos.move_one_fibre_m.measure(self.expose,'mos','fib')
        else:
            print("Failed to close jaws... powering off and stopping!")
            plc_controller.robot_power(robot,0)
            return
                
        mf.open_jaws()
        if (mf.wait()):
            print((xpix-xrot)*scale,(ypix-yrot)*scale)
        else:
            print("Failed to open jaws... powering off and stopping!")
            plc_controller.robot_power(robot,0)
            return
            
        mf.close_jaws()
        if (mf.wait()):
            if (robot == 1):
                xpix1,ypix,xplate,yplate=pos.move_one_fibre_n.measure(self.expose,'mos','fib')
            else:
                xpix1,ypix,xplate,yplate=pos.move_one_fibre_m.measure(self.expose,'mos','fib')
        else:
            print("Failed to close jaws... powering off and stopping!")
            plc_controller.robot_power(robot,0)
            return

        mf.open_jaws()
        if (mf.wait()):
            print((xpix1-xrot)*scale,(ypix-yrot)*scale)
        else:
            print("Failed to open jaws... powering off and stopping!")
            plc_controller.robot_power(robot,0)
            return
        mf.close_jaws()
        if (mf.wait()):
            if (robot == 1):
                xpix2,ypix,xplate,yplate=pos.move_one_fibre_n.measure(self.expose,'mos','fib')
            else:
                xpix2,ypix,xplate,yplate=pos.move_one_fibre_m.measure(self.expose,'mos','fib')
        else:
            print("Failed to close jaws... powering off and stopping!")
            plc_controller.robot_power(robot,0)
            return
        mf.open_jaws()
        if (mf.wait()):
            f=open('grasp_offsets.txt',mode='a+')
            print((xpix2-xrot)*scale,(ypix-yrot)*scale)
            grasp_x = scale*((xpix1+xpix2)/2.-xrot)
            s=str(self.plate)+' '+str(self.fibid)+' '+str(robot)+' '+str(grasp_x)+'\n'
            f.write(s)
            f.close()
        else:
            print("Failed to open jaws... powering off and stopping!")
            plc_controller.robot_power(robot,0)
            return
        
        mf.move_z(25.)
        mf.wait()
        if (parkit):
            mf.move_z(0.)
            mf.wait()
            self.nudge(robot,-2,0)
            pos.park_fibre(robot,self.fibid)
        return(grasp_x)

    def find_grasp_guide(self,robot,parkit=True):
        self.zpark=self.parkz_read(robot,self.fibid)
        xx,yy=self.safe_pull_out(130)
        pos.unpark_fibre(robot,self.fibid,xx,yy)
#        self.move_over_fibre_plate(robot)
        myp=Position(0.,0.,0.)
        mf=Movement(robot,myp)
        mf.move_z(self.zvalue(xx,yy,robot)[0]+self.plate_release)
        mf.wait()
        time.sleep(1.)
        mf.close_jaws()
        mf.wait()
        if (robot == 1):
            self.gexpose=pos_config.get('fibre_measure.exposure_time.nona.guide',
                                       float)
            if (self.plate == POSLIB.PLATE_A):
                _dict = pos_config.get_dict(['fibre_measure','exposure_time','nona','plate_a'])
                try:
                    self.gexpose = float(_dict[str(self.fibid)])
                except:
                    print("No predefined exposure set, using default")
            else:
                _dict = pos_config.get_dict(['fibre_measure','exposure_time','nona','plate_b'])
                try:
                    self.gexpose = float(_dict[str(self.fibid)])
                except:
                    print("No predefined exposure set, using default")
            xrot=pos_config.get('cameras.gripper_n.fibre.rotation_axis.x',
                                       float)
            yrot=pos_config.get('cameras.gripper_n.fibre.rotation_axis.y',
                                       float)
            scale=pos_config.get('cameras.gripper_n.fibre.pixel_scale',
                                       float)
            xpix,ypix,xplate,yplate=pos.move_one_fibre_n.measure(self.gexpose,'guide','fib')
        else:
            self.gexpose=pos_config.get('fibre_measure.exposure_time.morta.guide',
                                       float)
            if (self.plate == POSLIB.PLATE_A):
                _dict = pos_config.get_dict(['fibre_measure','exposure_time','morta','plate_a'])
                try:
                    self.gexpose = float(_dict[str(self.fibid)])
                except:
                    print("No predefined exposure set, using default")
            else:
                _dict = pos_config.get_dict(['fibre_measure','exposure_time','morta','plate_b'])
                try:
                    self.gexpose = float(_dict[str(self.fibid)])
                except:
                    print("No predefined exposure set, using default")
            xrot=pos_config.get('cameras.gripper_m.fibre.rotation_axis.x',
                                       float)
            yrot=pos_config.get('cameras.gripper_m.fibre.rotation_axis.y',
                                       float)
            scale=pos_config.get('cameras.gripper_m.fibre.pixel_scale',
                                       float)
            xpix,ypix,xplate,yplate=pos.move_one_fibre_m.measure(self.gexpose,'guide','fib')
        mf.open_jaws()
        mf.wait()
        print((xpix-xrot)*scale,(ypix-yrot)*scale)
        mf.close_jaws()
        mf.wait()
        if (robot == 1):
            xpix1,ypix,xplate,yplate=pos.move_one_fibre_n.measure(self.gexpose,'guide','fib')
        else:
            xpix1,ypix,xplate,yplate=pos.move_one_fibre_m.measure(self.gexpose,'guide','fib')
        mf.open_jaws()
        mf.wait()
        print((xpix1-xrot)*scale,(ypix-yrot)*scale)
        mf.close_jaws()
        mf.wait()
        if (robot == 1):
            xpix2,ypix,xplate,yplate=pos.move_one_fibre_n.measure(self.gexpose,'guide','fib')
        else:
            xpix2,ypix,xplate,yplate=pos.move_one_fibre_m.measure(self.gexpose,'guide','fib')
        mf.open_jaws()
        mf.wait()
        f=open('grasp_offsets.txt',mode='a+')
        print((xpix2-xrot)*scale,(ypix-yrot)*scale)
        grasp_x = scale*((xpix1+xpix2)/2.-xrot)
        s=str(self.plate)+' '+str(self.fibid)+' '+str(robot)+' '+str(grasp_x)+'\n'
        f.write(s)
        f.close()
        mf.move_z(30.)
        mf.wait()
        if (parkit):
            pos.park_fibre(robot,self.fibid)
        return(grasp_x)

    def update_single_grasp_entry(self,robot,graspx):
        if (robot == 0):
            self.offsetfile=pos_config.get_filename('morta.gripper_offset_map')
        else:
            self.offsetfile=pos_config.get_filename('nona.gripper_offset_map')
        backupfile=self.offsetfile+'_backup'
        if (self.plate == POSLIB.PLATE_A):
            myplate=0
        else:
            myplate=1
        self.zz=np.loadtxt(self.offsetfile,dtype=np.float64)
        fibs = np.asarray(self.zz.T[1],dtype=np.int32)
        index = np.where(fibs == self.fibid)[0][myplate]
        oldvalue = self.zz[index,2]
        np.savetxt(backupfile,self.zz,fmt='%d %d %6.3f %6.3f %6.3f %6.3f %7.3f %6.3f %6.3f %6.3f %6.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')
        self.zz[index,2] = -graspx
        np.savetxt(self.offsetfile,self.zz,fmt='%d %d %6.3f %6.3f %6.3f %6.3f %7.3f %6.3f %6.3f %6.3f %6.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')
        if (robot == 0):
            pos.move_one_fibre_m.offline()
            pos.move_one_fibre_m.online()
        else:
            pos.move_one_fibre_n.offline()
            pos.move_one_fibre_n.online()
        print("Grasp Offset updated for fibre ", self.fibid," robot ", robot,": ",-graspx)
        print("Old value was ",oldvalue)
        return


    def update_grasp_files(self,grasplogfile='grasp_offsets.txt'):
        moffsetfile = pos_config.get_filename('morta.gripper_offset_map')
        noffsetfile = pos_config.get_filename('nona.gripper_offset_map')
        newmoffsetfile = '/home/pos_eng/WEAVE/etc/morta_gripper_offset_map.txt_new'
        newnoffsetfile = '/home/pos_eng/WEAVE/etc/nona_gripper_offset_map.txt_new'
            
        zzm=np.loadtxt(moffsetfile,dtype=np.float64)
        zzn=np.loadtxt(noffsetfile,dtype=np.float64)
        offsets=np.loadtxt(grasplogfile,dtype=np.str)
        for i in offsets:
            _plate = i[0]
            myplate = 0
            if (_plate == 'PLATE_B'):
                myplate = 1
            fib = int(i[1])
            robot = int(i[2])
            grasp = -float(i[3])
            if (robot == 0):
                zzm[fib*2+myplate,2] = grasp
            else:
                zzn[fib*2+myplate,2] = grasp
        np.savetxt(newmoffsetfile,zzm,fmt='%d %d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')            
        np.savetxt(newnoffsetfile,zzn,fmt='%d %d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')            
        return

    def old_update_release_file(self,releaselogfile='releasoffsets.txt'):
        zz=np.loadtxt(self.offsetfile,dtype=np.float64)
        fibs=np.asarray(zz.T[1],dtype=np.int32)
        if (self.plate == POSLIB.PLATE_A):
            myplate = 0
        else:
            myplate = 1
        offsets=np.loadtxt(releaselogfile,dtype=np.str)
        xx = offsets.T[12]
        yy = offsets.T[14]
        myfibs = offsets.T[2]
        frompark= offsets.T[4]
        robot = offsets.T[8]
        topark=offsets.T[10]
        for i in range(0,np.shape(xx)[0],1):
            myfib = int(re.split(',',myfibs[i])[0])
            index = np.where(fibs == myfib)[0][myplate]
            gx = zz[index,2]
            gy = zz[index,3]
            mydx = float(re.split(',',xx[i])[0])+gx
            mydy = float(re.split('}',yy[i])[0])+gy
            if (frompark[i] == 'True,'):
                zz[index,7] = -mydx
                zz[index,8] = -mydy
                print("Unpark: ",myfib,-mydx,-mydy)
            elif (topark[i] == 'True,'):
                zz[index,9] = -mydx
                zz[index,10] = -mydy
                print("Park: ",myfib,-mydx,-mydy)
            else:
                zz[index,4] = -mydx
                zz[index,5] = -mydy
                print("Plate: ",myfib,-mydx,-mydy)
        newfile='new_offset_file.txt'
        np.savetxt(newfile,zz,fmt='%d %d %6.3f %6.3f %6.3f %6.3f %7.3f %6.3f %6.3f %6.3f %6.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')
        return

    def parse_release_file(self,releaselogfile='releasoffsets.txt'):
        print('Using Grasp offset from ',self.offsetfile)
        zz=np.loadtxt(self.offsetfile,dtype=np.float64)
        fibs=np.asarray(zz.T[1],dtype=np.int32)
        if (self.plate == POSLIB.PLATE_A):
            myplate = 0
        else:
            myplate = 1
        offsets=np.loadtxt(releaselogfile,dtype=np.str)
        xx = offsets.T[12]
        yy = offsets.T[14]
        myfibs = offsets.T[2]
        frompark= offsets.T[4]
        robot = offsets.T[8]
        topark=offsets.T[10]
        px=[]
        py=[]
        for i in range(0,np.shape(xx)[0],1):
            myfib = int(re.split(',',myfibs[i])[0])
            index = np.where(fibs == myfib)[0][myplate]
            gx = zz[index,2]
            gy = zz[index,3]
            mydx = float(re.split(',',xx[i])[0])+gx
            mydy = float(re.split('}',yy[i])[0])+gy
            if (frompark[i] == 'True,'):
                zz[index,7] = -mydx
                zz[index,8] = -mydy
#                print(myfib,-mydx,-mydy," Unpark")
            elif (topark[i] == 'True,'):
                zz[index,9] = -mydx
                zz[index,10] = -mydy
#                print(myfib,-mydx,-mydy," Park")
            else:
                zz[index,4] = -mydx
                zz[index,5] = -mydy
#                print(myfib,-mydx,-mydy," Move")
                if (myfib == self.fibid):
                    px.append(-mydx)
                    py.append(-mydy)
#        newfile='new_offset_file.txt'
#        np.savetxt(newfile,zz,fmt='%d %d %6.3f %6.3f %6.3f %6.3f %7.3f %6.3f %6.3f %6.3f %6.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')
        return(np.asarray(px,dtype=np.float64),np.asarray(py,dtype=np.float64))


    def parse_park_release_file(self,releaselogfile='releasoffsets.txt'):
        print('Using Grasp offset from ',self.offsetfile)
        zz=np.loadtxt(self.offsetfile,dtype=np.float64)
        fibs=np.asarray(zz.T[1],dtype=np.int32)
        if (self.plate == POSLIB.PLATE_A):
            myplate = 0
        else:
            myplate = 1
        offsets=np.loadtxt(releaselogfile,dtype=np.str)
        xx = offsets.T[12]
        yy = offsets.T[14]
        myfibs = offsets.T[2]
        frompark= offsets.T[4]
        robot = offsets.T[8]
        topark=offsets.T[10]
        px=[]
        py=[]
        upx=[]
        upy=[]
        for i in range(0,np.shape(xx)[0],1):
            myfib = int(re.split(',',myfibs[i])[0])
            index = np.where(fibs == myfib)[0][myplate]
            gx = zz[index,2]
            gy = zz[index,3]
            mydx = float(re.split(',',xx[i])[0])+gx
            mydy = float(re.split('}',yy[i])[0])+gy
            if (frompark[i] == 'True,'):
                zz[index,7] = -mydx
                zz[index,8] = -mydy
#                print(myfib,-mydx,-mydy," Unpark")
                if (myfib == self.fibid):
                    upx.append(-mydx)
                    upy.append(-mydy)
            elif (topark[i] == 'True,'):
                zz[index,9] = -mydx
                zz[index,10] = -mydy
#                print(myfib,-mydx,-mydy," Park")
                if (myfib == self.fibid):
                    px.append(-mydx)
                    py.append(-mydy)
            else:
                zz[index,4] = -mydx
                zz[index,5] = -mydy
#                print(myfib,-mydx,-mydy," Move")
#        newfile='new_offset_file.txt'
#        np.savetxt(newfile,zz,fmt='%d %d %6.3f %6.3f %6.3f %6.3f %7.3f %6.3f %6.3f %6.3f %6.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')
        return(np.asarray(upx,dtype=np.float64),np.asarray(upy,dtype=np.float64),np.asarray(px,dtype=np.float64),np.asarray(py,dtype=np.float64))



    def patchuppark(self,robot,max_pos_error=0.5,qmode=True,ddx=0,ddy=0,nudge=False):
        # go to park position go gently down to height and take an image, but don't try to store it.
        # If it's in range for pos.move_fibre, leave it alone
        # If it's behind where it should be, or left/right, pick it up as is, budge it, and put it down again then re-check.
        # If it's ahead of where it should be, move to pick it up where it is and then park it.
        _current = pos.fibre_position_store.get_fibre(self.plate, self.fibid)
        self.max_pos_error=max_pos_error
        _x = _current.start_x
        _y = _current.start_y
        dx = _x - self.xpiv
        dy = _y - self.ypiv
        self.zpark = self.parkz_read(robot,self.fibid)
        th=np.arctan2(dy,dx)
        _t = self.range_angle((np.pi + th)/self.fac,robot)
        myp=Position(0.,0.,0.)
        mf=Movement(robot,myp)
        mf.move_z(self.clear_z) # move to above retractors
        mf.wait()
        self.move_over_park_plate(robot)
        mf.move_z(self.zpark-4.0) # move to just above the vane
        mf.wait()
        if (nudge):
            self.nudge(robot,ddx,ddy)
        if (qmode):
            pos.plc_controller.gripper_spotlight_on(robot)
            myinput=input("OK to move down?")
            if (myinput == "n" or myinput == "N"):
                return
            pos.plc_controller.gripper_spotlight_off(robot)
        mf.move_z(self.zpark)
        time.sleep(0.2)
        test = mf.wait()
        if (not test):
            print ("seem to have hit something...")
            return()
        mf.close_jaws()
        mf.wait()
        mf.open_jaws()
        mf.wait()
        #Now we should be at the correct angle, so the position is valid
        _xx,_yy,_zz,_tt,_flag=pos.plc_controller.get_robot_position(robot)
        expose=self.expose
        if (self.image_type=="guide"):
            expose=self.gexpose
        if (robot == 0):
            a,b,xx,yy = pos.move_one_fibre_m.measure(expose,self.image_type,self.focus)
        else:
            a,b,xx,yy = pos.move_one_fibre_n.measure(expose,self.image_type,self.focus)
#            a,b,xx,yy = pos.move_one_fibre_n.measure(2.,'guide',self.focus)
        if (a == -1 or b == -1):
            print ("No fibre seen here.. suggest try ...measure() by hand?")
            return
        parkrad = np.sqrt(self.xpark*self.xpark + self.ypark*self.ypark)
        posrad = np.sqrt(xx*xx+yy*yy)
        dp = np.sqrt((xx-self.xpark)*(xx-self.xpark)+(yy-self.ypark)*(yy-self.ypark))
        if (dp < self.max_pos_error):
            print("Position is in range, ready to move out...")
            # stuff here for move_out action, or leave to the end?
        elif (posrad > parkrad): # we're further out, so can't move to pick up
                delta_x = self.xpark - xx
                delta_y = self.ypark - yy
                mf.close_jaws()
                if (mf.wait()):
                    print("Nudging forwards")
                    self.nudge(robot, 0., 0., dz=-1.0) # lift
                    self.nudge(robot, delta_x,delta_y) # tweak
                    self.nudge(robot, 0., 0., dz=1.0) # lower
                    mf.open_jaws()
                    if (mf.wait()):
                # We should now be within tolerance of position and have
                # not broken anything we hadn't broken already
                        print("Nudged it forward to park position, ready to move out")
                    else:
                        print("Failed to open jaws... powering off and stopping!")
                        pos.plc_controller.robot_power(robot,0)
                        return
                else:
                    print("Failed to close jaws... powering off and stopping!")
                    pos.plc_controller.robot_power(robot,0)
                    return
        else:
            # we're further forwards than we should be, so need to move the robot by the difference (inwards) and put back...
                delta_x = self.xpark - xx
                delta_y = self.ypark - yy
                print("Repositioning robot")
                self.nudge(robot, 0.,0.,dz=-4.0) # move clear
                self.nudge(robot, -delta_x, -delta_y) # shift robot position
                self.nudge(robot, 0., 0., dz=4.0)
                mf.close_jaws()
                if (mf.wait()):
                    print("Nudging Backwards")
                    self.nudge(robot, 0., 0., dz=-1.0) # lift
                    self.nudge(robot, delta_x, delta_y) # tweak
                    self.nudge(robot, 0., 0., dz=1.0) # lower
                    mf.open_jaws()
                    if (mf.wait()):
                        print ("Nudged it back to park position, ready to move out")
                    else:
                        print("Failed to open jaws... powering off and stopping!")
                        plc_controller.robot_power(robot,0)
                        return
                else:
                    print("Failed to close jaws... powering off and stopping!")
                    plc_controller.robot_power(robot,0)
                    return
# All done here... now we need to move clear in case the software decides to
# pirouette for show...
        mf.move_z(self.clear_z)
        mf.wait()
# comment these out for now 
#        XX,YY = self.safe_pull_out(150)
#        pos.unpark_fibre(robot, self.fibid, XX, YY)
#        mf.move_z(self.clear_z)
#        mf.wait()
#        pos.park_fibre(robot, self.fibid)
        return


    def record_zpark2(self,robot,zdti):
        if (robot == 0):
            offsetfile=pos_config.get_filename('morta.gripper_offset_map')
            offsetfile_other=pos_config.get_filename('nona.gripper_offset_map')
        else:
            offsetfile=pos_config.get_filename('nona.gripper_offset_map')
            offsetfile_other=pos_config.get_filename('morta.gripper_offset_map')
        if (self.plate == POSLIB.PLATE_A):
            myplate=0
        else:
            myplate=1
        zz=np.loadtxt(offsetfile,dtype=np.float64)
        zz2=np.loadtxt(offsetfile_other,dtype=np.float64)
        mynew_zpark = -(zdti-28.26)-5.0
        if (self.plate == POSLIB.PLATE_A):
            myplate=0
        else:
            myplate=1
        old_val = zz[self.fibid*2+myplate,6]
        mydiff = mynew_zpark-old_val
        old_val_2 = zz2[self.fibid*2+myplate,6]
        mynew_zpark2 = old_val_2 + mydiff+0.3 # empirical offset
        zz[self.fibid*2+myplate,6] = mynew_zpark
        zz[(self.fibid-1)*2+myplate,6] = mynew_zpark+10.
        zz[(self.fibid-2)*2+myplate,6] = mynew_zpark+20.
        zz2[self.fibid*2+myplate,6] = mynew_zpark2
        zz2[(self.fibid-1)*2+myplate,6] = mynew_zpark2+10.
        zz2[(self.fibid-2)*2+myplate,6] = mynew_zpark2+20.
        newname=offsetfile+'_update2'
        np.savetxt(offsetfile,zz,fmt='%d %d %6.3f %6.3f %6.3f %6.3f %7.3f %6.3f %6.3f %6.3f %6.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')
        np.savetxt(offsetfile_other,zz2,fmt='%d %d %6.3f %6.3f %6.3f %6.3f %7.3f %6.3f %6.3f %6.3f %6.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')
        return


    def gen_exercise(self,num_pos,max_theta=11.0):
        # assume fibre is starting in park. Use pos commands for moves.
        # Make a set of moves for the fibre, ending back in park, and assume we're logging the offsets.
        r1 = np.sqrt(np.concatenate(np.random.rand(1,np.int32(num_pos)),axis=None))
        pp = np.concatenate(np.random.uniform(-1,1,size=(1,num_pos)),axis=None)
        pp = (max_theta*pp)*self.fac

        min_length = self.min_length + 35.
        max_length = self.max_length - 25.

        r1 = self.min_length + r1*(max_length-min_length)
        ang1 = np.arctan2(-self.ypiv,-self.xpiv)/self.fac
        ang2 = np.arctan2(self.ypark-self.ypiv,self.xpark-self.xpiv)/self.fac
        da = ang1 - ang2
        angle = 180.+da+ np.arctan2((self.ypark-self.ypiv),(self.xpark-self.xpiv))/self.fac
        tt = angle*self.fac + pp
        xx = (self.xpiv - r1*np.cos(tt))
        yy = (self.ypiv - r1*np.sin(tt))
        rad = np.sqrt(xx*xx+yy*yy)
        psi = np.arctan2(yy,xx)
        self.cone_xx = xx
        self.cone_yy = yy
        self.valid = np.full_like(xx,dtype=np.bool_,fill_value=True)
        _sx=self.start.start_x
        _sy=self.start.start_y
        _st=self.start.start_t
        for j in range (0,np.shape(xx)[0],1):
            dx = xx[j]-self.xpiv
            dy = yy[j]-self.ypiv
            th = np.arctan2(dy,dx)/self.fac
            try:
                print(self.plate,self.fibid,_sx,_sy,_st,xx[j],yy[j],th)
                pos.fibre_position_store.move(self.plate,self.fibid,_sx,_sy,_st,xx[j],yy[j],th)
            except:
                self.valid[j] = False
            pos.fibre_position_store.cancel_move(self.plate,self.fibid)
            #            self.cone_xx[j],self.cone_yy[j]=pos.move_one_fibre_m.robot_to_plate(xx[j],yy[j])
        self.gen_flag = True
        if (self.valid[0] == False):
            for j in range(np.shape(xx)[0]-1,0,-1):
                if (self.valid[j]):
                    self.cone_xx[0] = self.cone_xx[j]
                    self.cone_yy[0] = self.cone_yy[j]
                    self.valid[0] = self.valid[j]
                    break
        return

    def run_exercise(self,robot,qmode=False,sleep=0.3):
        xp=[]
        yp=[]
        ts=[]
        self.zpark = self.parkz_read(robot,self.fibid) # this will force us to use the right offsets file!
        print('Using Grasp offset from ',self.offsetfile)
        if (self.gen_flag == False):
            print("No points to run around!")
            return
        if (qmode):
            myinput1="u"
        else:
            myinput1=input("Enter u to unpark, g to find grasp, c if already out on the plate?")
        if (myinput1 == "u" or myinput1 == "g" or myinput1 == "c"):
            if (myinput1 == "u"):
                print ("Unparking to ",self.cone_xx[0],self.cone_yy[0])
                pos.unpark_fibre(robot,self.fibid,self.cone_xx[0],self.cone_yy[0])
                _x,_y,_z,_t,_flag = pos.plc_controller.get_robot_position(robot)
                if (_x > 0):
                    self.nudge(robot,-0.5,0.0,dz=-0.5)
                else:
                    self.nudge(robot,0.5,0.0,dz=-0.5)
            elif (myinput1 == "g"):
                if (self.image_type == "guide"):
                    graspx = self.find_grasp_guide(robot,parkit=False)
                else:
                    graspx = self.find_grasp(robot,parkit=False)
                self.update_single_grasp_entry(robot,graspx)
            if (qmode):
                myinput2 = "y"
                myinput4 = "a"
            else:
                myinput2 = input("Enter y to proceed?")
                myinput4 = "n"
            if (myinput2 == "y" or myinput2 == "Y"):
                for j in range (1,np.shape(self.cone_xx)[0],1):
                    if (self.valid[j]):
                        print("moving to ",self.cone_xx[j],self.cone_yy[j])
                        pos.move_fibre(robot,self.fibid,self.cone_xx[j],self.cone_yy[j])
                        _x,_y,_z,_t,_flag = pos.plc_controller.get_robot_position(robot)
                        if (_x > 0):
                            self.nudge(robot,-0.5,0.0,dz=-0.5)
                        else:
                            self.nudge(robot,0.5,0.0,dz=-0.5)
                        xp.append(self.cone_xx[j])
                        yp.append(self.cone_yy[j])
                        ts.append(datetime.now().isoformat())
                        time.sleep(sleep)
                        if (myinput4 != "a" and myinput4 != "A"):
                            myinput4 = input("Enter a for all, n for next, z to exit loop?")
                            if (myinput4 == "z" or myinput4 == "Z"):
                                break
            if (qmode):
                myinput3 = "p"
            else:
                myinput3 = input("Enter p to park, q to quit?")
            if (myinput3 == "p" or myinput3 == "P"):
                pos.park_fibre(robot,self.fibid)
                _x,_y,_z,_t,_flag = pos.plc_controller.get_robot_position(robot)
                if (_x > 0):
                    self.nudge(robot,-0.5,0.0,dz=-0.5)
                else:
                    self.nudge(robot,0.5,0.0,dz=-0.5)
            px,py=self.parse_release_file("offsetlogging.txt")
            _xp=np.asarray(xp,dtype=np.float64)
            npt = _xp.shape[0]
            _out = np.zeros(npt,dtype=[('xp','float'),('yp','float'),('xe','float'),('ye','float'),('ts','U26')])
            _out['xp'] = np.asarray(xp,dtype=np.float64)
            _out['yp'] = np.asarray(yp,dtype=np.float64)
            _out['xe'] = px[-npt:]
            _out['ye'] = py[-npt:]
            _out['ts'] = np.asarray(ts,dtype='U26')
            fname='offsets/move.'+str(robot)+str(self.plate)+str(self.fibid)
            f=open(fname,'ab')
            np.savetxt(f,_out,fmt='%11.3f %11.3f %11.3f %11.3f %26s')
            f.close()
        return

    def measure_rotation_centre(self,robot,ntries=10):
        if (robot == 0):
            move_one_fibre_me = pos.move_one_fibre_m
            thmax = pos_config.get('morta.theta_axis.high_limit',float)
            thmin = pos_config.get('morta.theta_axis.low_limit',float)
        else:
            move_one_fibre_me = pos.move_one_fibre_n
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
        xx,yy=self.safe_pull_out(140)
        pos.unpark_fibre(robot,self.fibid,xx,yy)
        zp = self.zvalue(xx,yy,robot)[0]+self.plate_release
        _x,_y,_z,_t,_flag = pos.plc_controller.get_robot_position(robot)
        if (robot == 1): # shift the fibre a bit to bring both images in field
            xx,yy = self.safe_pull_out(140.5)
            self.nudge(robot,0.,0.,dz=-25.)
            pos.move_fibre(robot,self.fibid,xx,yy)
        diff1 = thmax - _t
        diff2 = _t - thmin
        if (diff1 > 180.):
            _t2 = _t + 180.
        else:
            _t2 = _t - 180.
        xp1 = []
        yp1 = []
        xp2 = []
        yp2 = []
        mypf=FullPosition(_x,_y,_t,zp) # go back to image height for the first
        mf=Movement(robot,mypf,speed=self.speed)
        mf.move_all()
        mf.wait()
        for i in range(0,ntries):
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(100,'mos','fib')
            xp1.append(_xp)
            yp1.append(_yp)
            print (_xp,_yp)
            mypf=FullPosition(_x,_y,_t,24.) # move to z=24
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            mypf=FullPosition(_x,_y,_t2,0.) # move to z=0 and rotate
            mf=Movement(robot,mypf,speed=5)
            mf.move_all()
            mf.wait()
            mypf=FullPosition(_x,_y,_t2,zp) # move to image
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(100,'mos','fib')
            xp2.append(_xp)
            yp2.append(_yp)
            print (_xp,_yp)
            mypf=FullPosition(_x,_y,_t2,24.) # move to z=24
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            mypf=FullPosition(_x,_y,_t,0.) # move to z=0 and rotate back
            mf=Movement(robot,mypf,speed=5)
            mf.move_all()
            mf.wait()
            mypf=FullPosition(_x,_y,_t,zp) # move to image
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
        mypf=FullPosition(_x,_y,_t,24.) # move to z=24
        mf=Movement(robot,mypf,speed=self.speed)
        mf.move_all()
        mf.wait()
        xp1n = np.array(xp1)
        yp1n = np.array(yp1)
        xp2n = np.array(xp2)
        yp2n = np.array(yp2)
        rx = (xp1n+xp2n)/2.
        ry = (yp1n+yp2n)/2.
        print (rx,ry)
        print (rx.mean(),ry.mean())
        pos.park_fibre(robot,self.fibid)
        return

    def measure_plate_camera_rotation_centre2(self,robot,fid_number=178,ntries=5):
        if (robot == 0):
            move_one_fibre_me = pos.move_one_fibre_m
            park_robot = pos.park_morta
            thmax = pos_config.get('morta.theta_axis.high_limit',float)
            thmin = pos_config.get('morta.theta_axis.low_limit',float)
            xdef = pos_config.get('cameras.gripper_m.plate.rotation_axis.x',float)
            ydef = pos_config.get('cameras.gripper_m.plate.rotation_axis.y',float)
            scalex = pos_config.get('cameras.gripper_m.plate.pixel_scale_x',float)
            scaley = pos_config.get('cameras.gripper_m.plate.pixel_scale_y',float)
        else:
            move_one_fibre_me = pos.move_one_fibre_n
            park_robot = pos.park_nona
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
            xdef = pos_config.get('cameras.gripper_n.plate.rotation_axis.x',float)
            ydef = pos_config.get('cameras.gripper_n.plate.rotation_axis.y',float)
            scalex = pos_config.get('cameras.gripper_n.plate.pixel_scale_x',float)
            scaley = pos_config.get('cameras.gripper_n.plate.pixel_scale_y',float)
        _t = 0.
        _t2 = -180.
        if (thmin > _t2):
            _t2 = 180.
        xr1 = []
        yr1 = []
        xr2 = []
        yr2 = []
        gx=pos.plate_info(self.plate).fiducials[fid_number].x
        gy=pos.plate_info(self.plate).fiducials[fid_number].y
        _x = gx+33.6
        _x1= gx-33.6
        _y = gy
        zp = self.zvalue(_x,_y,robot)[0]+pos_config.get('z_axis.image_plate', float)+0.6
        mypf=FullPosition(_x,_y,_t,zp) # move to starting point
        mf=Movement(robot,mypf,speed=self.speed)
        mf.move_all()
        mf.wait()
        time.sleep(0.5)
        pos.plc_controller.gripper_spotlight_on(robot)
        for i in range(0,ntries):
            _xp,_yp,_xcalc,_ycalc=move_one_fibre_me.measure(50,'fid','plate')
            basex = pos_config.get('plate_measure.camera_pixel.x',float)
            basey = pos_config.get('plate_measure.camera_pixel.y',float)
            dxp = (_xp-basex)*scaley
            dyp = (_yp-basey)*scalex
            print (dxp,dyp)
            self.nudge(robot,-dyp,-dxp)
            time.sleep (0.5) # spot should now be on basex,basey
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(50,'fid','plate')
            xx1,yy1,zz1,tt1,_flag=pos.plc_controller.get_robot_position(robot)
            xr1.append(xx1)
            yr1.append(yy1)
            print (xx1,yy1,_xp,_yp,_t)
            zp1 = self.zvalue(_x1,_y,robot)[0]+pos_config.get('z_axis.image_plate', float)
            mypf=FullPosition(_x1,_y,_t2,zp1) # move to opposite position
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            time.sleep(0.5)
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(50,'fid','plate')
            dxp = (_xp-basex)*scaley
            dyp = (_yp-basey)*scalex
            print (dxp,dyp)
            self.nudge(robot,dyp,dxp)
            time.sleep (0.5) # spot should now be on basex,basey
            xx2,yy2,zz2,tt2,_flag=pos.plc_controller.get_robot_position(robot)
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(50,'fid','plate')
            print(xx2,yy2,_xp,_yp,_t2)
            xr2.append(xx2)
            yr2.append(yy2)
            mypf=FullPosition(xx1,yy1,tt1,zz1) # move back to start
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            time.sleep(0.5)
# Now we have all the measurements
        xp1n = np.array(xr1)
        yp1n = np.array(yr1)
        xp2n = np.array(xr2)
        yp2n = np.array(yr2)
        rx = (xp2n-xp1n)/2.
        ry = (yp2n-yp1n)/2.
        print (rx,ry)
        print (rx[1:].mean(),ry[1:].mean())
        dx = ry[1:].mean()/scalex+basex
        dy = rx[1:].mean()/scaley+basey
        print(ry[1:].mean()/scalex,rx[1:].mean()/scaley)
        print('Starting values', xdef,ydef)
        # Now we just need to figure out which way to apply
        print('Suggested new values ',dx,dy)
        pos.plc_controller.gripper_spotlight_off(robot)
        park_robot()
        return dx,dy

    def measure_plate_camera_rotation_centre(self,robot,fid_number=178,ntries=5):
        if (robot == 0):
            move_one_fibre_me = pos.move_one_fibre_m
            park_robot = pos.park_morta
            thmax = pos_config.get('morta.theta_axis.high_limit',float)
            thmin = pos_config.get('morta.theta_axis.low_limit',float)
            xdef = pos_config.get('cameras.gripper_m.plate.rotation_axis.x',float)
            ydef = pos_config.get('cameras.gripper_m.plate.rotation_axis.y',float)
            myscale = pos_config.get('cameras.gripper_m.plate.pixel_scale',float)
        else:
            move_one_fibre_me = pos.move_one_fibre_n
            park_robot = pos.park_nona
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
            xdef = pos_config.get('cameras.gripper_n.plate.rotation_axis.x',float)
            ydef = pos_config.get('cameras.gripper_n.plate.rotation_axis.y',float)
            myscale = pos_config.get('cameras.gripper_n.plate.pixel_scale',float)
        _t = 0.
        _t2 = -180.
        if (thmin > _t2):
            _t2 = 180.
        xp1 = []
        yp1 = []
        xp2 = []
        yp2 = []
        gx=pos.plate_info(self.plate).fiducials[fid_number].x
        gy=pos.plate_info(self.plate).fiducials[fid_number].y
        _x = gx+33.6
        _x1= gx-33.6
        _y = gy
        zp = self.zvalue(_x,_y,robot)[0]+pos_config.get('z_axis.image_plate', float)+0.6
        mypf=FullPosition(_x,_y,_t,zp) # move to starting point
        mf=Movement(robot,mypf,speed=self.speed)
        mf.move_all()
        mf.wait()
        time.sleep(0.5)
        pos.plc_controller.gripper_spotlight_on(robot)
        for i in range(0,ntries):
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(50,'fid','plate')
            xp1.append(_xx-gx)
            yp1.append(_yy-gy)
            print (_xx-gx,_yy-gy)
            zp1 = self.zvalue(_x1,_y,robot)[0]+pos_config.get('z_axis.image_plate', float)+0.6
            mypf=FullPosition(_x1,_y,_t2,zp1) # move to opposite position
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            time.sleep(0.5)
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(50,'fid','plate')
            print(_xx-gx,_yy-gy)
            xp2.append(_xx-gx)
            yp2.append(_yy-gy)
            mypf=FullPosition(_x,_y,_t,zp) # move back to start
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            time.sleep(0.5)
# Now we have all the measurements
        xp1n = np.array(xp1)
        yp1n = np.array(yp1)
        xp2n = np.array(xp2)
        yp2n = np.array(yp2)
        rx = (xp2n-xp1n)/2.
        ry = (yp2n-yp1n)/2.
        print (rx,ry)
        print (rx[1:].mean(),ry[1:].mean())
        dx = ry[1:].mean()/myscale
        dy = rx[1:].mean()/myscale
        print('Starting values', xdef,ydef)
        # Now we just need to figure out which way to apply
        print('Changes ',dx,dy)
        if (robot == 0):
            xnew = xdef+dx
            ynew = ydef+dy
        else:
            xnew = xdef+dx
            ynew = ydef+dy
        print('New values', xnew,ynew)
        pos.plc_controller.gripper_spotlight_off(robot)
        park_robot()
        return xnew,ynew

    def measure_plate_camera_rotation_centre_fibre(self,robot,fib_number,ntries=5):
        if (robot == 0):
            move_one_fibre_me = pos.move_one_fibre_m
            park_robot = pos.park_morta
            thmax = pos_config.get('morta.theta_axis.high_limit',float)
            thmin = pos_config.get('morta.theta_axis.low_limit',float)
            xdef = pos_config.get('cameras.gripper_m.plate.rotation_axis.x',float)
            ydef = pos_config.get('cameras.gripper_m.plate.rotation_axis.y',float)
            myscale = pos_config.get('cameras.gripper_m.plate.pixel_scale',float)
        else:
            move_one_fibre_me = pos.move_one_fibre_n
            park_robot = pos.park_nona
            thmax = pos_config.get('nona.theta_axis.high_limit',float)
            thmin = pos_config.get('nona.theta_axis.low_limit',float)
            xdef = pos_config.get('cameras.gripper_n.plate.rotation_axis.x',float)
            ydef = pos_config.get('cameras.gripper_n.plate.rotation_axis.y',float)
            myscale = pos_config.get('cameras.gripper_n.plate.pixel_scale',float)
        _t = 0.
        _t2 = -180.
        if (thmin > _t2):
            _t2 = 180.
        xp1 = []
        yp1 = []
        xp2 = []
        yp2 = []
        gx = pos.fibre_position_store.get_fibre(self.plate,fib_number).start_x
        gy = pos.fibre_position_store.get_fibre(self.plate,fib_number).start_y
        _x = gx+33.6
        _x1= gx-33.6
        _y = gy
        zp = self.zvalue(_x,_y,robot)[0]+pos_config.get('z_axis.image_plate', float)-1.6
        mypf=FullPosition(_x,_y,_t,zp) # move to starting point
        mf=Movement(robot,mypf,speed=self.speed)
        mf.move_all()
        mf.wait()
        time.sleep(0.5)
        for i in range(0,ntries):
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(100,'mos','plate')
            xp1.append(_xx-gx)
            yp1.append(_yy-gy)
            print (_xx-gx,_yy-gy)
            zp1 = self.zvalue(_x1,_y,robot)[0]+pos_config.get('z_axis.image_plate', float)-1.6
            mypf=FullPosition(_x1,_y,_t2,zp1) # move to opposite position
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            time.sleep(0.5)
            _xp,_yp,_xx,_yy = move_one_fibre_me.measure(100,'mos','plate')
            print(_xx-gx,_yy-gy)
            xp2.append(_xx-gx)
            yp2.append(_yy-gy)
            mypf=FullPosition(_x,_y,_t,zp) # move back to start
            mf=Movement(robot,mypf,speed=self.speed)
            mf.move_all()
            mf.wait()
            time.sleep(0.5)
# Now we have all the measurements
        xp1n = np.array(xp1)
        yp1n = np.array(yp1)
        xp2n = np.array(xp2)
        yp2n = np.array(yp2)
        rx = (xp2n-xp1n)/2.
        ry = (yp2n-yp1n)/2.
        print (rx,ry)
        print (rx[1:].mean(),ry[1:].mean())
        dx = ry[1:].mean()/myscale
        dy = rx[1:].mean()/myscale
        print('Starting values', xdef,ydef)
        # Now we just need to figure out which way to apply
        print('Changes ',dx,dy)
        if (robot == 0):
            xnew = xdef+dx
            ynew = ydef+dy
        else:
            xnew = xdef+dx
            ynew = ydef+dy
        print('New values', xnew,ynew)
        park_robot()
        return xnew,ynew

    def run_dti_grid(self,robot,destfile='junk.txt',filename='dti_positions.txt',lower=True,speed=30):
        points = np.loadtxt(filename,dtype=np.float)
        mypos = FullPosition(float(points[0,0]),float(points[0,1]),self.range_angle(float(points[0,3]),robot),0.)
        z0=input("Tell me your z value where you zeroed the DTI?")
        zd=input("Tell me your ZD value?")
        wrs=input("Tell me the WRS angle?")
        print("moving to ",mypos)
        mf=Movement(robot,mypos,speed)
        mf.move_all()
        mf.wait()
        dti_values = []
        if (lower):
            mypos = FullPosition(float(points[0,0]),float(points[0,1]),self.range_angle(float(points[0,3]),robot),float(points[0,2]))
            mf=Movement(robot,mypos,speed)
            mf.move_all()
        mf.wait()
        j=-1
        for k in points:
            j=j+1
            if (lower):
                z=float(points[j,2])
            else:
                z=0.
            mypos=FullPosition(float(points[j,0]),float(points[j,1]),self.range_angle(float(points[j,3]),robot),z)
            print("moving to ",mypos)
            print("Point number ",j," of ",np.shape(points)[0])
            mf=Movement(robot,mypos,speed)
            mf.move_all()
            mf.wait()
            try:
                points[j,2] =  float(input('Enter DTI Value'))
            except:
                print('Doh!... that did not look like a number.. one more try...?')
                points[j,2] =  float(input('Enter DTI Value'))
        print('Finished... writing file')
        mystring = str(self.plate)+" "+str(robot)+" "+str(zd)+" "+str(wrs)+" "+str(z0)
        np.savetxt(destfile,points,fmt=' %8.3f ',header=mystring)
        return

    def fixit(self,robot):
        # pickup from a robot that has landed on a vane... sort out the park position and move on
        pos.positioner.offline()
        pos.positioner.online()
        foo = pos.plate_info(self.plate).fibres[self.fibid]
        _x,_y,_z,_t,_flag = pos.plc_controller.get_robot_position(robot)
        rad2 = np.sqrt(_x*_x+_y*_y)
        dz = self.parkz_read(robot,self.fibid)-_z
        if ((dz < 0) or (dz > 3)):
            print("too far off park height to be a simple fix.. exiting")
            return False
        if (foo.type == "Spectrograph"):
            mytype = "mos"
            myexpos = self.expose
        else:
            mytype = "guide"
            myexpos = self.gexpose
        if (robot == 0):
            _a,_b,_x,_y = pos.move_one_fibre_m.measure(myexpos,mytype,"fibre")
        else:
            _a,_b,_x,_y = pos.move_one_fibre_n.measure(myexpos,mytype,"fibre")
        rad3 = np.sqrt(_x*_x+_y*_y)
        dx = _x - self.xpark
        dy = _y - self.ypark
        if (rad3 <= rad2+0.5):
            self.patchuppark(robot,max_pos_error=0.2,qmode=False,nudge=True,ddx=dx,ddy=dy)
        else:
            rad4 = np.sqrt(_x*_x+self.ypark*self.ypark)
            if (rad4 > rad2+0.5):
                rad5 = np.sqrt(self.xpark*self.xpark+_y*_y)
                self.patchuppark(robot,max_pos_error=0.2,qmode=False,nudge=True,ddx=0.,ddy=dy)
            else:
                self.patchuppark(robot,max_pos_error=0.2,qmode=False,nudge=True,ddx=dx,ddy=0.)
        return True

            

    def run_park_exercise(self,robot,qmode=False,sleep=0.3):
        xp=[]
        yp=[]
        ts=[]
        self.zpark = self.parkz_read(robot,self.fibid) # this will force us to use the right offsets file!
        print('Using Grasp offset from ',self.offsetfile)
        if (self.gen_flag == False):
            print("No points to run around!")
            return
        if (qmode):
            myinput2 = "y"
            myinput4 = "a"
        else:
            myinput2 = input("Enter y to proceed?")
            myinput4 = "n"
        if (myinput2 == "y" or myinput2 == "Y"):
            for j in range (0,np.shape(self.cone_xx)[0],1):
                if (self.valid[j]):
                    print("unparking to ",self.cone_xx[j],self.cone_yy[j])
                    pos.unpark_fibre(robot,self.fibid,self.cone_xx[j],self.cone_yy[j])
                    _x,_y,_z,_t,_flag = pos.plc_controller.get_robot_position(robot)
                    if (_x > 0):
                        self.nudge(robot,-0.5,0.0,dz=-0.5)
                    else:
                        self.nudge(robot,0.5,0.0,dz=-0.5)
                    xp.append(self.cone_xx[j])
                    yp.append(self.cone_yy[j])
                    ts.append(datetime.now().isoformat())
                    if (qmode):
                        myinput3 = "p"
                        myinput4 = "a"
                    else:
                        if (myinput4 != "a" and myinput4 != "A"):
                            myinput3 = input("Enter p to park, q to quit?")
                    if (myinput3 == "p" or myinput3 == "P"):
                        pos.park_fibre(robot,self.fibid)
                        _x,_y,_z,_t,_flag = pos.plc_controller.get_robot_position(robot)
                        if (_x > 0):
                            self.nudge(robot,-0.5,0.0,dz=-0.5)
                        else:
                            self.nudge(robot,0.5,0.0,dz=-0.5)
                    if (myinput4 != "a" and myinput4 != "A"):
                        myinput4 = input("Enter a for all, n for next, z to exit loop?")
                        if (myinput4 == "z" or myinput4 == "Z"):
                            break
        _x,_y,_z,_t,_flag = pos.plc_controller.get_robot_position(robot)
        self.nudge(robot,0,0,-_z)
        upx,upy,px,py=self.parse_park_release_file("offsetlogging.txt")
        _xp=np.asarray(xp,dtype=np.float64)
        npt = _xp.shape[0]
        _outu = np.zeros(npt,dtype=[('xp','float'),('yp','float'),('xe','float'),('ye','float'),('ts','U26')])
        _outp = np.zeros(npt,dtype=[('xp','float'),('yp','float'),('xe','float'),('ye','float'),('ts','U26')])
        _outp['xp'] = np.asarray(xp,dtype=np.float64)[-npt:]
        _outp['yp'] = np.asarray(yp,dtype=np.float64)[-npt:]
        _outp['xe'] = px[-npt:]
        _outp['ye'] = py[-npt:]
        _outp['ts'] = np.asarray(ts,dtype='U26')[-npt:]
        _outu['xp'] = np.asarray(xp,dtype=np.float64)[-npt:]
        _outu['yp'] = np.asarray(yp,dtype=np.float64)[-npt:]
        _outu['xe'] = upx[-npt:]
        _outu['ye'] = upy[-npt:]
        _outu['ts'] = np.asarray(ts,dtype='U26')[-npt:]
        fname='offsets/park.'+str(robot)+str(self.plate)+str(self.fibid)
        f=open(fname,'ab')
        np.savetxt(f,_outp,fmt='%11.3f %11.3f %11.3f %11.3f %26s')
        f.close()
        fname='offsets/unpark.'+str(robot)+str(self.plate)+str(self.fibid)
        f=open(fname,'ab')
        np.savetxt(f,_outu,fmt='%11.3f %11.3f %11.3f %11.3f %26s')
        f.close()
        return

    def update_release_file(self,releaselogfile='for_ingest'):
        moffsetfile = pos_config.get_filename('morta.gripper_offset_map')
        noffsetfile = pos_config.get_filename('nona.gripper_offset_map')
        newmoffsetfile = 'morta_gripper_offset_map.txt_new'
        newnoffsetfile = 'nona_gripper_offset_map.txt_new'
            
        zzm=np.loadtxt(moffsetfile,dtype=np.float64)
        zzn=np.loadtxt(noffsetfile,dtype=np.float64)
        print(moffsetfile, noffsetfile)
        offsets=np.loadtxt(releaselogfile,dtype=np.float64)
        for i in offsets:
            myplate = 0
            if (self.plate == POSLIB.PLATE_B):
                myplate = 1
            fib = int(i[0])
            robot = int(i[1])
            if (robot == 0):
                zzm[fib*2+myplate,4] = i[2]
                zzm[fib*2+myplate,5] = i[3]
                zzm[fib*2+myplate,7] = i[6]
                zzm[fib*2+myplate,8] = i[7]
                zzm[fib*2+myplate,9] = i[10]
                zzm[fib*2+myplate,10] = i[11]
               
            else:
                zzn[fib*2+myplate,4] = i[2]
                zzn[fib*2+myplate,5] = i[3]
                zzn[fib*2+myplate,7] = i[6]
                zzn[fib*2+myplate,8] = i[7]
                zzn[fib*2+myplate,9] = i[10]
                zzn[fib*2+myplate,10] = i[11]
                
        np.savetxt(newmoffsetfile,zzm,fmt='%d %d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')            
        np.savetxt(newnoffsetfile,zzn,fmt='%d %d %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f %7.3f',header='DATAMVER 8.00\n plate fibre pick_x pick_y place_x place_y park_z place_from_park_x place_from_park_y place_at_park_x place_at_park_y')            
        return
