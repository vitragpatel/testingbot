from firebase_admin import credentials, db
import firebase_admin


# Initialize Firebase Admin SDK
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(
    cred, {"databaseURL": "https://flaskapp-6b34b-default-rtdb.firebaseio.com/"}
)


def get_site_user_data():
    ref = db.reference("site-user")
    return ref.get()


def get_hotel_data():
    ref = db.reference("hotels/hotels")
    return ref.get()