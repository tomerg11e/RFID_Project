# RFID_Projet

log in into the terminal through diagnostic- importent make sure the LAN is on. can be on while connecting the the thingmagic via the router while having the computer connecte to the router as well
username: debian  password rootsecure
first sudo su
now change the software time using the commend "ntpdate -u pool.ntp.org"
now update the hardware time using the comment "hwclock -w"
and now check with "hwclock" and "date" that the two are the same and the same as the real time

in etc/adjtime we can see the zero timestamp

in active/bin there is ther file that responsible for the time updating called updatetime.sh

usful commands:
systemctl status ntp.service for seeing the ntp status
ntpq -p for seeing all the ntp connections
going into the proc folder gives you an idea for the status of the threads and deamons currently running n the system.
by finding the process id we can go into proc/PROCESS_ID/task/PROCESS_ID and see the second line of the status file.
