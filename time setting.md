# Time Setting

##changing time manually
changing the time setting in the izar thingmagic:
<br/>
connect the thingmagic and your computer to the same router (not switch!), 
now log in to the izar thingmagic web client. make sure the LAN indicator is on.
<br/>
<br/>
Enter diagnostic and log into the terminal. <br/>
username: debian <br/>  password: rootsecure<br/>
type sudo su for root privilege. <br/>
change the software time using the command:
<br/> ```ntpdate -u {your wanted ntp server ip}```<br/>
and update the hardware time using the comment:<br/> ```hwclock -w```<br/>
Now, check with ```hwclock``` and ```date``` commands that the two times are equal and synced to the real time




##useful commands and files:
for seeing the ntp status use: ```systemctl status ntp.service``` <br/>
for seeing all the ntp connections use: ```ntpq -p``` <br/>
in etc/adjtime you can see the zero timestamp <br/>
going into the proc folder gives you an idea for the status of the threads and deamons currently running n the system.
by finding the process id we can go into proc/PROCESS_ID/task/PROCESS_ID and see the second line of the status file. <br/>
in active/bin there is the file that responsible for the time updating called updatetime.sh


using the update-time.sh we can see that for updating the rtc with the wanted ntp we need three conditions:
the ntp status need to be running
the ntp_servers variable need to not be "" - taken from /tm/etc/tm.conf (the same as written in the web settings)
the server_status variable need to not be "" - taken from the old code, 1 only if we can connect using http to our ntp server **LAN problematic** mabye even just problematic...

```console
if [ "${ntp_servers}" ]; then
    wget -q -O /tmp/foo ${ntp_servers} >/tmp/foo
    if [ $(grep 'http' /tmp/foo | wc -l) -ne 0 ]; then
        SERVER_STATUS=1
    fi
    rm -rf /tmp/foo
fi
```
now just trying to ping the ntp_server
