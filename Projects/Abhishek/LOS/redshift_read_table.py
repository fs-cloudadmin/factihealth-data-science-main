import redshift_connector
import pandas as pd

redshift_endpoint = (
    "redshift-cluster-factihealth.cuzgotkwtow6.ap-south-1.redshift.amazonaws.com"
)
redshift_dbname = "factihealth"
redshift_port = 5439
redshift_user = "fh_user"
redshift_pass = "Facti@874"

conn = redshift_connector.connect(
    host=redshift_endpoint,
    database=redshift_dbname,
    port=int(redshift_port),
    user=redshift_user,
    password=redshift_pass,
)


def read_table(query):

    cursor = conn.cursor()
    cursor.execute(query)
    rows = cursor.fetchall()
    column_names = [desc[0] for desc in cursor.description]

    dataframe = pd.DataFrame(rows, columns=column_names)

    cursor.close()

    return dataframe
