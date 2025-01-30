from datetime import datetime, timedelta
import json
from pathlib import Path
from collections import defaultdict
import argparse

def parse_timestamp(timestamp_str):
    """Parse timestamp with variable precision in fractional seconds."""
    try:
        return datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
    except ValueError:
        try:
            date_part, tz_part = timestamp_str.rsplit('-', 1)
            if '.' in date_part:
                base, frac = date_part.rsplit('.', 1)
                frac = frac.ljust(6, '0')
                date_part = f"{base}.{frac}"
            normalized_timestamp = f"{date_part}-{tz_part}"
            return datetime.fromisoformat(normalized_timestamp)
        except Exception as e:
            print(f"Error parsing timestamp '{timestamp_str}': {e}")
            raise

def get_time_bucket(timestamp, bucket_size_hours=4):
    """Get the time bucket for a timestamp."""
    hour = timestamp.hour
    bucket = (hour // bucket_size_hours) * bucket_size_hours
    return timestamp.replace(hour=bucket, minute=0, second=0, microsecond=0)

def chunk_chat_export(input_file, output_dir='chunked_chats', bucket_size_hours=4):
    """Process Discord chat export into time-bucketed chunks."""
    # Load the input file
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract channel info
    channel_info = {
        'id': data['channel']['id'],
        'name': data['channel'].get('name'),
        'topic': data['channel'].get('topic'),
        'category': data['channel'].get('category')
    }
    channel_info = {k: v for k, v in channel_info.items() if v is not None}
    
    # Create output directory
    channel_output_path = Path(output_dir) / str(channel_info['id'])
    channel_output_path.mkdir(parents=True, exist_ok=True)
    
    # Group messages by time buckets
    time_buckets = defaultdict(list)
    user_map = {}
    
    for message in data.get('messages', []):
        # Parse timestamp and get bucket
        ts = parse_timestamp(message['timestamp'])
        bucket = get_time_bucket(ts, bucket_size_hours)
        
        # Clean message and add to appropriate bucket
        if message.get('content'):
            # Add user to map if not exists
            user_id = message['author']['id']
            if user_id not in user_map:
                user_data = {}
                if name := message['author'].get('name'):
                    user_data['name'] = name
                if nickname := message['author'].get('nickname'):
                    user_data['nickname'] = nickname
                if roles := [r['name'] for r in message['author'].get('roles', []) if r.get('name')]:
                    user_data['roles'] = roles
                if message['author'].get('isBot'):
                    user_data['isBot'] = True
                user_map[user_id] = user_data
            
            # Clean message
            cleaned_msg = {
                'id': message['id'],
                'ts': message['timestamp'],
                'uid': user_id,
                'content': message['content']
            }
            
            # Add optional fields
            if message_type := message.get('type'):
                if message_type != 'Default':
                    cleaned_msg['type'] = message_type
            if edited := message.get('timestampEdited'):
                cleaned_msg['edited'] = edited
            if mentions := [m['id'] for m in message.get('mentions', []) if m.get('id')]:
                cleaned_msg['mentions'] = mentions
            if ref := message.get('reference', {}).get('messageId'):
                cleaned_msg['ref'] = ref
            if reactions := message.get('reactions'):
                cleaned_reactions = [{'emoji': r['emoji'].get('name'), 'count': r['count']} 
                                   for r in reactions]
                if cleaned_reactions:
                    cleaned_msg['reactions'] = cleaned_reactions
            
            time_buckets[bucket].append(cleaned_msg)
    
    # Write time-bucketed files
    files_created = 0
    for bucket, messages in sorted(time_buckets.items()):
        if messages:  # Only create files for buckets with messages
            bucket_str = bucket.strftime('%Y-%m-%d_%H%M')
            output_file = channel_output_path / f'chat_{bucket_str}.json'
            
            output_data = {
                'channel': channel_info,
                'date': bucket.strftime('%Y-%m-%d'),
                'timeBlock': f"{bucket.strftime('%H:%M')}-{(bucket + timedelta(hours=bucket_size_hours)).strftime('%H:%M')}",
                'users': user_map,
                'messages': messages
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            files_created += 1
    
    return files_created

def main():
    parser = argparse.ArgumentParser(description='Split Discord chat export into time-bucketed chunks')
    parser.add_argument('input_file', help='Input JSON file path')
    parser.add_argument('output_dir', help='Output directory for chunked files')
    parser.add_argument('--bucket-size', type=int, default=4, help='Size of time buckets in hours (default: 4)')
    
    args = parser.parse_args()
    
    num_files = chunk_chat_export(args.input_file, args.output_dir, args.bucket_size)
    print(f'Successfully created {num_files} time-bucketed chat files')

if __name__ == '__main__':
    main()
