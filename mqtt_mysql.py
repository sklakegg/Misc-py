#!/usr/bin/python
import time
import re
import paho.mqtt.client as mqtt
import pymysql.cursors

# The callback for when the client receives a CONNACK response from the server.
def on_connect(client, userdata, flags, rc):
    print("Connected with result code "+str(rc))

    # Subscribing in on_connect() means that if we lose the connection and
    # reconnect then subscriptions will be renewed.
    client.subscribe("#", 2)

# The callback for when a PUBLISH message is received from the server.
def on_message(client, userdata, msg):
    print(msg.topic+" "+str(msg.payload))
    epoch = float(time.time())
    result = re.search('#', msg.topic)
    devID = result.group(1)
    data = str(msg.payload)
    # Connect to the database.
    connection = pymysql.connect(host='#', user='#', password='#', db='#', charset='utf8mb4', cursorclass=pymysql.cursors.DictCursor)
    try:
    	with connection.cursor() as cursor:
            # Create a new record
            sql = "INSERT INTO `device_data` (`timestamp`, `device_id`, `sensor_data`) VALUES (%s, %s, %s)"
            cursor.execute(sql, (epoch, devID, data))

            # connection is not autocommit by default. So you must commit to save
            # your changes.
            connection.commit()

    finally:
        connection.close()

client = mqtt.Client()
client.on_connect = on_connect
client.on_message = on_message

client.connect("IP", 1883, 60)

# Blocking call that processes network traffic, dispatches callbacks and
# handles reconnecting.
# Other loop*() functions are available that give a threaded interface and a
# manual interface.
client.loop_forever()
