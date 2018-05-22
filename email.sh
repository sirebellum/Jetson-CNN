#!/bin/bash

today=`date +%Y-%m-%d.%H:%M:%S`

{
    echo To: bluedude227@gmail.com
    echo From: $HOSTNAME
    echo Subject: 'Notification!'
    echo
    echo $today

} | sudo ssmtp bluedude227@gmail.com
