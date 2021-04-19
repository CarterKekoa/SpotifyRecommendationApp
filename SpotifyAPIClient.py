from urllib.parse import urlencode
import requests
import datetime
import base64

class SpotifyAPI(object):
    access_token = None
    access_token_expires = datetime.datetime.now()
    access_token_did_expire = True
    client_id = None
    client_secret = None
    token_url = "https://accounts.spotify.com/api/token"
    
    def __init__(self, client_id, client_secret, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.client_id = client_id
        self.client_secret = client_secret

    def get_client_credentials(self):
        """
        Returns a base64 encoded string
        """
        client_id = self.client_id
        client_secret = self.client_secret
        if client_secret == None or client_id == None:
            raise Exception("You must set client_id and client_secret")
        client_creds = f"{client_id}:{client_secret}"
        client_creds_b64 = base64.b64encode(client_creds.encode())
        return client_creds_b64.decode()
    
    def get_token_headers(self):
        client_creds_b64 = self.get_client_credentials()
        return {
            "Authorization": f"Basic {client_creds_b64}"
        }
    
    def get_token_data(self):
        return {
            "grant_type": "client_credentials"
        } 
    
    def perform_auth(self):
        token_url = self.token_url
        token_data = self.get_token_data()
        token_headers = self.get_token_headers()
        r = requests.post(token_url, data=token_data, headers=token_headers)
        if r.status_code not in range(200, 299):
            raise Exception("Did not authenticate")
        data = r.json()
        now = datetime.datetime.now()
        access_token = data['access_token']
        expires_in = data['expires_in'] # seconds
        expires = now + datetime.timedelta(seconds=expires_in)
        self.access_token = access_token
        self.access_token_expires = expires
        self.access_token_did_expire = expires < now
        return True

    def get_access_token(self):
        token = self.access_token
        expires = self.access_token_expires
        now = datetime.datetime.now()
        if expires < now:
            self.perform_auth()
            return self.get_access_token()
        elif token == None:
            self.perform_auth()
            return self.get_access_token()
        return token

    def get_access_header(self):
        access_token = self.get_access_token()
        headers = {
            'Authorization': f"Bearer {access_token}"
        }
        return headers

    # Get Spotify Catalog information about albums, artists, playlists, tracks, shows or episodes that match a keyword string.
    # query: Key Word(string) ex: song name, albumn name or artist name
    # search_type: album , artist, playlist, track, show or episode (string)
    # limit: amount of responses(int)
    def search(self, query, search_type='track', limit=5):
        headers = self.get_access_header()
        endpoint = 'https://api.spotify.com/v1/search?'
        data = urlencode({
            'q': query,
            'type': search_type.lower(),
            'limit': limit
        })
        lookup_url = f"{endpoint}{data}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    def get_album(self, album_id):
        base_url = 'https://api.spotify.com/v1'
        endpoint = f'{base_url}/albums/{album_id}'
        headers = self.get_access_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    def get_user_playlist(self, user_id, limit=5, offset=0):
        base_url = 'https://api.spotify.com/v1/users'
        endpoint = f'{base_url}/{user_id}/playlists?'
        headers = self.get_access_header()
        data = urlencode({
            'limit': limit,
            'offset': offset 
        })
        lookup_url = f"{endpoint}{data}"
        r = requests.get(lookup_url, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    def get_playlist(self, playlist_id):
        base_url = 'https://api.spotify.com/v1/playlists'
        endpoint = f'{base_url}/{playlist_id}'
        headers = self.get_access_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    def parse_for_track_id(self, response):
        track_ids = {}
        for val in response['tracks']['items']:
            track_name = (val['track']['name'])
            track_id = (val['track']['id'])
            track_ids[track_name] = track_id
        return track_ids

    def get_track_audio_features(self, track_id):
        base_url = 'https://api.spotify.com/v1/audio-features'
        endpoint = f'{base_url}/{track_id}'
        headers = self.get_access_header()
        r = requests.get(endpoint, headers=headers)
        if r.status_code not in range(200, 299):
            return {}
        return r.json()

    